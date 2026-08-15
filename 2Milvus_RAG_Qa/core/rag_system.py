"""RAG 系统核心 — 意图识别 → 策略选择 → 检索 → LLM 生成。

完整链路(A2 整改后,BERT 已退出主链路):
    用户 query
      → 意图识别(无额外开销规则短路 + LLM 分类)
          ├─ 闲聊 → 直接 LLM 对话
          └─ 知识检索 → 策略选择器 → 双路召回(求职意图)/混合检索 → LLM 生成
"""

import sys
import time
from pathlib import Path
from typing import Any, Iterator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI

from base.config import cfg
from base.logger import logger
from .prompts import RAGPrompts
from .rag_cache import RagCache
from .strategy_selector import StrategySelector
from .vector_store import VectorStore
from .dual_retrieval import dual_retrieve, extract_jd_conditions, should_trigger_jd


def _prompt_version() -> str:
    """当前 prompt 版本(懒加载,避免与 agent_graph 的循环导入)。"""
    try:
        from .agent_graph import PROMPT_VERSION
        return PROMPT_VERSION
    except Exception:
        return "unknown"


class RAGSystem:
    """RAG 检索增强生成系统。

    功能：
        - 意图识别：无额外开销规则短路(寒暄)+ LLM 分类(BERT 已退出主链路,见 A2)
        - 四种检索策略：直接/假设/子问题/回溯(规则映射优先,LLM 兜底)
        - 双路召回(求职意图:JD 结构化 + 向量)/ 混合检索(稠密+稀疏→RRF→精排→MMR)→ LLM 生成
    """

    # 检索置信度拒答阈值：向量路重排分 top1 低于此值直接拒答
    # (JD 结构化槽位 score=1.0 为置顶标记,不参与拒答判定)
    REJECT_THRESHOLD: float = 0.15
    REJECT_MESSAGE: str = "抱歉，知识库中未找到相关资料"

    # RAG 检索策略 → 意图类别映射
    RAG_INTENTS = {"课程咨询", "技术问题", "概念解释", "学习方法", "就业薪资", "项目实战"}

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._vs = vector_store or VectorStore()
        self._llm = OpenAI(api_key=cfg.LLM_API_KEY, base_url=cfg.LLM_BASE_URL, timeout=120)
        self._model = cfg.LLM_MODEL
        self._strategy_selector = StrategySelector()
        self._prompts = RAGPrompts()
        self._cache = RagCache()
        # 注(A2 整改):BERT 意图分类器已退出主链路,不再加载

    # ──────────────── 主入口 ────────────────

    def query(self, user_query: str, verbose: bool = False,
              session_id: str | None = None, memory_hint: str = "") -> dict:
        """RAG 问答主入口。

        Args:
            user_query: 用户原始问题
            verbose:    是否输出详细过程（调试用）
            session_id: 会话 ID；None 表示单轮（不加载历史、不写历史），
                非 None 时启用多轮指代消解与会话历史（D11）

        Returns:
            {"answer": ..., "sources": [...], "strategy": ..., "intent": ...,
             "latency_ms": ..., "from_cache": ...}
        """
        t0 = time.perf_counter()

        # 0. 多轮指代消解（有历史时用 LLM 轻量改写，关 reasoning）
        history = self._cache.get_history(session_id) if session_id else []
        resolved_query = self._resolve_coreference(user_query, history) if history else user_query
        if verbose and resolved_query != user_query:
            print(f"[改写] {resolved_query}")

        # 1. 意图识别
        intent = self._classify_intent(resolved_query)
        if verbose:
            print(f"[意图] {intent}")

        # 2. 闲聊直接 LLM 回复
        if intent in ("闲聊/其他", "chitchat"):
            answer = self._chat(resolved_query)
            result = self._result(answer, [], "闲聊", intent, t0)
            self._cache.append_history(session_id, user_query, answer)
            return result

        # 3. 语义级响应缓存（D12：相似 query 直接返回缓存答案）
        cached = self._try_semantic_cache(resolved_query, intent)
        if cached is not None:
            cached["latency_ms"] = round((time.perf_counter() - t0) * 1000)
            cached["from_cache"] = True
            self._cache.append_history(session_id, user_query, cached.get("answer", ""))
            return cached

        # 4. 策略选择（规则映射优先，未命中回退 LLM）
        strategy = self._strategy_selector.rule_based_select(intent) or \
            self._strategy_selector.select_strategy(resolved_query)
        if verbose:
            print(f"[策略] {strategy}")

        # 5. 检索 + 置信度拒答(整改:拒答只看向量路 top1 重排分,
        #    JD 结构化槽位(score=1.0 置顶标记)不参与,避免异构分数混判)
        scored = self._retrieve_scored(strategy, resolved_query, intent=intent)
        if verbose:
            print(f"[召回] {len(scored)} 条文档")
        vec_top = self._top_vector_score(scored)
        if not scored or (vec_top is None or vec_top < self.REJECT_THRESHOLD):
            result = self._result(self.REJECT_MESSAGE, [], strategy, intent, t0)
            self._cache.append_history(session_id, user_query, self.REJECT_MESSAGE)
            return result

        contexts = [d["text"] for d in scored]

        # 6. 上下文分级组装（长上下文时代:top 块原文 + 其余压缩补充,替代硬截断）
        contexts = self._vs.build_layered_contexts(resolved_query, contexts)
        if memory_hint:
            contexts = [memory_hint] + contexts

        # 7. RAG 生成
        answer = self._generate(resolved_query, contexts)
        result = self._result(answer, contexts, strategy, intent, t0)

        # 8. 写语义缓存 + 会话历史
        self._write_semantic_cache(resolved_query, intent, strategy, answer, contexts)
        self._cache.append_history(session_id, user_query, answer)
        return result

    # ──────────────── 意图识别 ────────────────

    def _classify_intent(self, query: str) -> str:
        """意图分类:无额外开销规则短路(寒暄)→ LLM 路由(BERT 已退场)。

        架构决策(A2):agentic 下路由需要"抽条件+决策循环",BERT 只能粗分类
        且实测频繁不置信回退 LLM 形同虚设,故移除 BERT 主链路,只留规则短路。
        """
        if self._is_chitchat_short(query):
            return "闲聊/其他"
        return self._llm_classify_intent(query)

    @staticmethod
    def _is_chitchat_short(query: str) -> bool:
        """无额外开销规则短路:纯寒暄/表情直接判闲聊,不调任何模型。"""
        q = query.strip().lower()
        if len(q) <= 10:
            for w in ("你好", "谢谢", "在吗", "hi", "hello", "嗨", "哈喽",
                      "早上好", "晚上好", "晚安", "再见", "好的", "ok"):
                if q.startswith(w):
                    return True
        return False

    def _llm_classify_intent(self, query: str) -> str:
        """大模型意图分类（BERT 不置信时的兜底）—— 输出短标签。

        max_tokens 提到 64：deepseek 推理模型会先产出 reasoning 再产出 content，
        16 token 会被 reasoning 吃满导致 content 100% 空返回（实测验证）。
        """
        prompt = self._prompts.intent_classify_prompt().format(question=query)
        return self._call_llm_with_retry(
            prompt,
            temperature=0.1,
            max_tokens=64,
            reasoning=None,
            fallback="知识检索",
        )

    # ──────────────── 四种检索策略 ────────────────

    def _retrieve_scored(
        self, strategy: str, query: str, intent: str | None = None
    ) -> list[dict]:
        """根据策略执行检索,返回 [{"text","score","source"}] 列表。

        S4:求职意图下"直接检索"升级为双路并行(JD 结构化 + 向量),合并文档。
        整改:返回 dict 并带 source 标记,使拒答/门控能区分
        JD 结构化槽位(score=1.0 为置顶标记)与向量重排分(异构分数不混判)。
        """
        if strategy == "直接检索":
            docs = dual_retrieve(
                query,
                self._vs.hybrid_search_with_rerank_scored,
                cfg.RETRIEVAL_K,
                intent=intent,
            )
            return docs

        if strategy == "假设问题检索":
            return self._retrieve_hyde_scored(query)

        if strategy == "子问题检索":
            return self._retrieve_subquery_scored(query)

        if strategy == "回溯问题检索":
            return self._retrieve_backtrack_scored(query)

        return self._retrieve_vector_only_scored(query)

    def _retrieve_vector_only_scored(self, query: str) -> list[dict]:
        """纯向量检索(带 source 标记),供非直接检索策略与兜底路径复用。"""
        return [
            {"text": t, "score": float(s), "source": "vector"}
            for t, s in self._vs.hybrid_search_with_rerank_scored(
                query, top_k=cfg.RETRIEVAL_K
            )
        ]

    @staticmethod
    def _top_vector_score(scored: list[dict]) -> float | None:
        """向量路 top1 重排分(JD 结构化槽位不参与,分数口径统一)。"""
        vec = [d.get("score", 0.0) for d in scored if d.get("source") != "jd"]
        return float(max(vec)) if vec else None

    def _retrieve_hyde_scored(self, query: str) -> list[dict]:
        """HyDE：LLM 生成假答案 → 用假答案检索。"""
        hyde_answer = self._ask_llm(
            self._prompts.hyde_prompt().format(question=query)
        )
        return self._retrieve_vector_only_scored(hyde_answer)

    def _retrieve_subquery_scored(self, query: str) -> list[dict]:
        """子问题检索：拆解 → 分别检索 → 父块去重合并（分数取最高）。"""
        subs_text = self._ask_llm(
            self._prompts.subquery_prompt().format(question=query)
        )
        sub_queries = [s.strip() for s in subs_text.split("\n") if s.strip()]

        best: dict[str, dict] = {}
        for sq in sub_queries[:4]:  # 最多 4 个子问题
            for doc in self._retrieve_vector_only_scored(sq):
                key = doc["text"]
                if key not in best or doc["score"] > best[key]["score"]:
                    best[key] = doc

        scored = sorted(best.values(), key=lambda x: x["score"], reverse=True)
        return scored[:cfg.RETRIEVAL_K + 1]

    def _retrieve_backtrack_scored(self, query: str) -> list[dict]:
        """回溯检索：简化为基础问题 → 检索。"""
        basic_q = self._ask_llm(
            self._prompts.backtrack_prompt().format(question=query)
        )
        return self._retrieve_vector_only_scored(basic_q.strip())

    # ──────────────── LLM 调用封装 ────────────────

    def _build_rag_prompt(self, query: str, contexts: list[str]) -> str:
        """拼接 RAG 生成 prompt,上下文每段带 [编号] 供答案引用(D9)。

        优化一(JD 隔离围栏):JD 结构化命中以"【招聘岗位】"开头,
        用显式围栏标记"仅用于求职类问题,严禁作为技术知识依据"——
        防止求职查询的 JD 内容被当成代码/技术原理写进技术答案(毒丸)。
        """
        if contexts:
            parts = []
            for i, ctx in enumerate(contexts):
                if ctx.strip().startswith("【招聘岗位】"):
                    parts.append(
                        f"[{i + 1}] 【招聘信息围栏:仅用于求职/薪资类问题,"
                        f"严禁作为代码/技术原理依据】\n{ctx}"
                    )
                else:
                    parts.append(f"[{i + 1}] {ctx}")
            ctx_text = "\n\n".join(parts)
        else:
            ctx_text = "（无检索上下文）"
        return self._prompts.rag_answer_prompt().format(
            context=ctx_text, question=query
        )

    def _generate(self, query: str, contexts: list[str]) -> str:
        """RAG 生成:reasoning high 质量优先,空返回/失败自动降级 low 兜底。

        max_tokens=4096:上下文约 2000 字时 reasoning 会占用大量预算,
        1024 会吃满导致 content 空返回(实测长 prompt 下空返回率显著升高)。
        延迟实测:high 模式长上下文一次生成约 30-40s(检索仅 ~0.5s),
        降级链 high→low 平衡"推理深度 vs 可用性";线上另有流式输出
        (用户先看到 token)与语义缓存(命中即省生成)缓解延迟。
        """
        prompt = self._build_rag_prompt(query, contexts)
        answer = self._call_llm_with_retry(
            prompt,
            max_tokens=4096,
            reasoning="high",
            fallback="",
        )
        if answer:
            return answer
        logger.warning("生成 reasoning=high 失败,降级 reasoning=low 重试")
        return self._call_llm_with_retry(
            prompt,
            max_tokens=2048,
            reasoning="low",
            fallback="抱歉，服务暂时不可用，请稍后重试。",
        )

    def _chat(self, query: str) -> str:
        """纯 LLM 对话（闲聊模式）。"""
        return self._ask_llm(query)

    def _resolve_coreference(self, query: str, history: list[dict[str, str]]) -> str:
        """多轮指代消解：结合历史把指代改写为明确实体（D11，关 reasoning）。"""
        history_text = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')}"
            for m in history
        )
        prompt = self._prompts.coreference_prompt().format(
            history=history_text, question=query
        )
        resolved = self._call_llm_with_retry(
            prompt,
            temperature=0.1,
            max_tokens=128,
            reasoning=None,
            fallback=query,
        )
        return resolved.strip() or query

    def _try_semantic_cache(self, query: str, intent: str) -> dict | None:
        """查询语义级响应缓存(D12),命中返回完整 result,未命中返回 None。

        整改(意图一致性校验):近似 query 但诉求不同会误命中——
        "北京 java 20k 岗位" 与 "上海 java 20k 岗位" 语义相似却要不同答案。
        因此缓存查找额外校验 ①意图一致 ②求职类硬槽(city/tech/salary)一致,
        否则视为未命中。
        """
        try:
            query_emb = self._vs.encode_query(query)
        except Exception as e:
            logger.warning("语义缓存 query 编码失败，跳过缓存: %s", e)
            return None
        slots = extract_jd_conditions(query) if should_trigger_jd(query, intent=intent) else {}
        payload = self._cache.get_semantic(
            query, query_emb, intent=intent, slots=slots,
            prompt_version=_prompt_version(), llm_model=self._model,
        )
        if payload is None:
            return None
        return {
            "answer": payload.get("answer", ""),
            "sources": payload.get("sources", []),
            "strategy": payload.get("strategy", "直接检索"),
            "intent": payload.get("intent", intent),
            "latency_ms": 0,
        }

    def _write_semantic_cache(
        self,
        query: str,
        intent: str,
        strategy: str,
        answer: str,
        sources: list[str],
    ) -> None:
        """写入语义级响应缓存(D12),Redis 不可用或编码失败则静默跳过。

        整改:payload 记 intent + 求职类硬槽,key 记 prompt/model 版本,
        语料版本戳由 RagCache 统一维护(重建自增 → 全量失效)。
        """
        try:
            query_emb = self._vs.encode_query(query)
        except Exception as e:
            logger.warning("语义缓存 query 编码失败，跳过写入: %s", e)
            return
        slots = extract_jd_conditions(query) if should_trigger_jd(query, intent=intent) else {}
        self._cache.set_semantic(query, query_emb, {
            "answer": answer,
            "sources": sources,
            "intent": intent,
            "strategy": strategy,
            "slots": slots,
            "prompt_version": _prompt_version(),
            "llm_model": self._model,
        })

    def _ask_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        """闲聊兜底 / 检索辅助小任务的 LLM 调用（reasoning low）。"""
        return self._call_llm_with_retry(
            prompt,
            temperature=0.3,
            max_tokens=max_tokens,
            reasoning="low",
            fallback="抱歉，服务暂时不可用，请稍后重试。",
        )

    def _call_llm_with_retry(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        reasoning: str | None = None,
        fallback: str,
    ) -> str:
        """LLM 调用统一封装：重试 3 次 + 空 content 视为失败。

        Args:
            prompt: 用户提示词
            temperature: 采样温度
            max_tokens: 最大输出 token 数
            reasoning: reasoning_effort 等级；None 时统一传 "low"
                （deepseek-v4-pro 不传 reasoning_effort / thinking disabled 会
                 100% 空返回 content，实测验证，故不再有"关闭推理"分支）
            fallback: 全部重试失败后的兜底返回

        Returns:
            LLM 返回文本（已 strip）；失败时返回 fallback
        """
        last_err: Exception | None = None
        for attempt in range(1, 4):  # deepseek 偶发空返回（~4%），重试 3 次
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    # deepseek-v4-pro 实测：不传 reasoning_effort / thinking disabled
                    # 时 100% 空返回 content，故 reasoning=None 也统一传 low
                    "reasoning_effort": reasoning or "low",
                }
                if reasoning is not None:
                    kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                resp = self._llm.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                last_err = ValueError("LLM 返回空 content")
            except Exception as e:
                last_err = e
            if attempt < 3:
                time.sleep(2 * attempt)
        logger.error("LLM 调用失败（重试 3 次）: %s", last_err)
        return fallback

    # ──────────────── 流式输出（E16）───────────────

    def _stream_llm(
        self,
        prompt: str,
        reasoning: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """流式 LLM 调用：逐 token yield 内容（reasoning 与 _call_llm_with_retry 一致）。"""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if reasoning is not None:
            kwargs["reasoning_effort"] = reasoning
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        try:
            stream = self._llm.chat.completions.create(**kwargs)
            for chunk in stream:
                if not chunk.choices:
                    continue
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    yield content
        except Exception as e:
            logger.error("流式 LLM 调用失败: %s", e)
            yield "抱歉，服务暂时不可用，请稍后重试。"

    def query_stream(
        self, user_query: str, session_id: str | None = None,
        memory_hint: str = "",
    ) -> Iterator[dict]:
        """流式问答（SSE 用），逐 token 产出事件 dict。

        事件格式：
            {"type": "token", "content": "..."}  增量文本
            {"type": "meta", "answer": ..., "intent": ..., "strategy": ...,
             "sources": [...], "latency_ms": ..., "from_cache": ...}  收尾元数据
        """
        t0 = time.perf_counter()
        history = self._cache.get_history(session_id) if session_id else []
        resolved_query = self._resolve_coreference(user_query, history) if history else user_query
        intent = self._classify_intent(resolved_query)

        def _meta(
            answer: str,
            sources: list[str],
            strategy: str,
            intent: str,
            from_cache: bool = False,
        ) -> dict:
            return {
                "type": "meta",
                "answer": answer,
                "intent": intent,
                "strategy": strategy,
                "sources": sources,
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "from_cache": from_cache,
            }

        # 闲聊 → 流式 LLM 对话
        if intent in ("闲聊/其他", "chitchat"):
            parts: list[str] = []
            for token in self._stream_llm(resolved_query, reasoning="low"):
                parts.append(token)
                yield {"type": "token", "content": token}
            answer = "".join(parts)
            self._cache.append_history(session_id, user_query, answer)
            yield _meta(answer, [], "闲聊", intent)
            return

        # 语义缓存命中 → 一次性返回缓存答案
        cached = self._try_semantic_cache(resolved_query, intent)
        if cached is not None:
            answer = cached.get("answer", "")
            yield {"type": "token", "content": answer}
            self._cache.append_history(session_id, user_query, answer)
            yield _meta(
                answer,
                cached.get("sources", []),
                cached.get("strategy", "直接检索"),
                intent,
                from_cache=True,
            )
            return

        # 策略 + 检索 + 拒答(整改:拒答只看向量路 top1 重排分)
        strategy = self._strategy_selector.rule_based_select(intent) or \
            self._strategy_selector.select_strategy(resolved_query)
        scored = self._retrieve_scored(strategy, resolved_query, intent=intent)
        vec_top = self._top_vector_score(scored)
        if not scored or (vec_top is None or vec_top < self.REJECT_THRESHOLD):
            answer = self.REJECT_MESSAGE
            yield {"type": "token", "content": answer}
            self._cache.append_history(session_id, user_query, answer)
            yield _meta(answer, [], strategy, intent)
            return

        contexts = [d["text"] for d in scored]
        contexts = self._vs.build_layered_contexts(resolved_query, contexts)
        if memory_hint:
            contexts = [memory_hint] + contexts

        prompt = self._build_rag_prompt(resolved_query, contexts)
        parts = []
        for token in self._stream_llm(prompt, reasoning="high"):
            parts.append(token)
            yield {"type": "token", "content": token}
        answer = "".join(parts)
        self._write_semantic_cache(resolved_query, intent, strategy, answer, contexts)
        self._cache.append_history(session_id, user_query, answer)
        yield _meta(answer, contexts, strategy, intent)

    # ──────────────── 结果封装 ────────────────

    @staticmethod
    def _result(
        answer: str, contexts: list[str], strategy: str, intent: str, t0: float
    ) -> dict:
        return {
            "answer": answer,
            "sources": contexts,
            "strategy": strategy,
            "intent": intent,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        }
