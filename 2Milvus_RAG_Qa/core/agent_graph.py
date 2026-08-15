# -*- coding: utf-8 -*-
"""LangGraph Adaptive RAG 工作流(最小闭环):检索 → 自省 → (改写 → 再检索) → 生成。

定位(2026-08-15 整改):**控制层 / Adaptive workflow,不是检索提分器**——
配对消融(McNemar,agent_compare.json)显示循环不提升纯单轮检索命中率;
价值 = 求职意图路由(双路召回)、失败改写自救、三级门控成本优化、异常兜底。

与经典 RAGSystem.query 的区别:
  - 经典:检索一次 → 直接生成(检索质量差时无法补救)
  - Adaptive:检索 → 自省"够不够" → 不够则改写 query 再检索 → 够则生成;
    改写次数用尽仍不充分 → 低置信度降级(明示证据不足、严禁编造)。

复用:
  - VectorStore.hybrid_search_with_rerank_scored  (检索)
  - RAGSystem._generate / _call_llm_with_retry    (生成 / LLM 调用)
不修改 RAGSystem,可与其做 A/B 对比。
"""
import json
import re
import time
from typing import TypedDict

from langgraph.graph import StateGraph, END

from base.config import cfg
from base.logger import logger

MAX_REWRITES = 2
# 优化七:级联仲裁三级门控(替代单一 0.6 阈值)
#   top1 ≥ 0.7 → 直接 pass(score_gate,无额外开销)
#   top1 ≤ 0.4 → 直接 fail(score_gate,不浪费 LLM 判明显没救的检索)
#   0.4 < top1 < 0.7(混沌区)→ LLM 仲裁(本地 0.5B 优先,回退云端)
# 收益:0.5B 只裁边界区,两端确定性区域 无额外开销;准确率/成本实测见 grade_quality.json
GRADE_SKIP_THRESHOLD = 0.7
GRADE_AUTO_FAIL = 0.4
PROMPT_VERSION = "agent-v3"  # P4:自省门控三级化 + JD 隔离围栏,版本随改动递增


class AgentState(TypedDict):
    query: str            # 当前问题(可能被改写)
    original_query: str   # 原始问题
    docs: list            # [{"text": str, "score": float}, ...]
    grade: str            # "pass" / "fail"
    grade_via: str        # 自省方式: score_gate / llm / no_docs
    rewrite_count: int    # 已改写次数
    answer: str
    memory_hint: str      # 记忆注入文本(画像+历史事实,可为空串)


class AdaptiveRAG:
    """LangGraph 驱动的自主检索循环,复用已初始化的 RAGSystem 实例。"""

    def __init__(self, rag_system, local_llm=None, tracer=None) -> None:
        self.rag = rag_system
        self.local = local_llm  # 快慢分层:None 时自省/改写走云端 deepseek
        self.tracer = tracer    # 可观测:每步事件 JSONL(可为 None)
        self._graph = self._build_graph()

    def _trace(self, event: str, **fields) -> None:
        """记录执行轨迹(可观测性,失败静默)。"""
        if self.tracer is not None:
            self.tracer.log(event, **fields)

    def _grade_llm(self, prompt: str) -> str:
        """自省判断:本地快模型优先,失败/空回退云端。"""
        if self.local is not None:
            v = self.local.generate(prompt, max_tokens=32)
            if v:
                return v
        return self.rag._call_llm_with_retry(
            prompt, temperature=0, max_tokens=512, reasoning=None, fallback="充分"
        )

    def _rewrite_llm(self, prompt: str) -> str:
        """改写:本地快模型优先,失败/空回退云端。"""
        if self.local is not None:
            v = self.local.generate(prompt, max_tokens=128)
            if v:
                return v.strip()
        return self.rag._ask_llm(prompt, max_tokens=512).strip()

    # ---------- 节点 ----------
    def _retrieve(self, state: AgentState) -> dict:
        """检索:求职意图双路并行(JD 结构化 + 向量),其余复用混合检索 + 重排。

        整改(改写只增不减):改写轮(retrieve_count>0)把本轮结果与上一轮
        按文本并集合并(同文本取最高分,JD 槽位保持最前,向量文档取 top_k)——
        配对消融实测首版"改写替换原文档"净损失 8 个命中(regression 10/rescue 2),
        并集合并后候选 ⊇ 首轮候选,改写只能加分不能减分。
        """
        t0 = time.time()
        from .dual_retrieval import dual_retrieve, merge_docs
        docs = dual_retrieve(
            state["query"],
            self.rag._vs.hybrid_search_with_rerank_scored,
            cfg.RETRIEVAL_K,
        )
        prev_docs = state.get("docs") or []
        if prev_docs:
            docs = merge_docs(prev_docs, docs, cfg.RETRIEVAL_K)
        self._trace(
            "retrieve", query=state["query"][:60], num_docs=len(docs),
            num_jd=sum(1 for d in docs if d.get("source") == "jd"),
            num_vector=sum(1 for d in docs if d.get("source") == "vector"),
            merged=bool(prev_docs),
            elapsed_ms=round((time.time() - t0) * 1000),
        )
        return {"docs": docs}

    def _grade(self, state: AgentState) -> dict:
        """自省(级联仲裁三级门控):两端确定性区域 无额外开销,混沌区 LLM 仲裁。

        整改(分数口径):门控分数只取 source=="vector" 的重排分 top1。
        JD 结构化命中(score=1.0)是独立槽位标记,与重排分异构,**不参与门控**——
        否则 JD 置顶会把门控永久拉到 1.0,求职类查询的自省形同虚设。
        纯 JD 命中(无向量文档)时按求职意图直接 pass,技术类查询则 fail。
        """
        t0 = time.time()
        docs = state.get("docs") or []
        if not docs:
            self._trace("grade", via="no_docs", elapsed_ms=round((time.time() - t0) * 1000))
            return {"grade": "fail", "grade_via": "no_docs"}
        vector_docs = [d for d in docs if d.get("source") != "jd"]
        if not vector_docs:
            # 仅 JD 结构化命中:求职意图=精确条件已满足,直接 pass;否则 fail
            has_jd = any(d.get("source") == "jd" for d in docs)
            if has_jd:
                self._trace("grade", via="jd_struct_only_pass",
                            elapsed_ms=round((time.time() - t0) * 1000))
                return {"grade": "pass", "grade_via": "jd_struct_only"}
            self._trace("grade", via="no_docs", elapsed_ms=round((time.time() - t0) * 1000))
            return {"grade": "fail", "grade_via": "no_docs"}
        top1 = max(float(d.get("score", 0.0)) for d in vector_docs)
        # 一级:向量 top1 ≥ 0.7 直接通过(省一次 LLM 调用)
        if top1 >= GRADE_SKIP_THRESHOLD:
            self._trace("grade", via="score_gate_pass", top1=round(top1, 3),
                        elapsed_ms=round((time.time() - t0) * 1000))
            return {"grade": "pass", "grade_via": "score_gate"}
        # 二级:向量 top1 ≤ 0.4 直接判不充分(明显没救,不浪费 LLM 调用)
        if top1 <= GRADE_AUTO_FAIL:
            self._trace("grade", via="score_gate_fail", top1=round(top1, 3),
                        elapsed_ms=round((time.time() - t0) * 1000))
            return {"grade": "fail", "grade_via": "score_gate"}
        # 三级:0.4 < top1 < 0.7 混沌区 → LLM 仲裁(本地快模型优先,回退云端)
        docs_text = "\n\n".join(
            f"[{i + 1}] {d['text'][:300]}" for i, d in enumerate(docs)
        )
        prompt = (
            "你是检索质量评估器。判断以下检索到的文档是否足以回答用户问题。\n"
            '只输出 JSON:{"verdict": "充分"} 或 {"verdict": "不充分"}。\n\n'
            "示例1:问题=Python多线程用法,文档=Python线程池用法详解 → {\"verdict\": \"充分\"}\n"
            "示例2:问题=Java协程用法,文档=JavaScript闭包原理 → {\"verdict\": \"不充分\"}\n\n"
            f"用户问题:{state['query']}\n\n"
            f"检索到的文档:\n{docs_text}\n"
        )
        # 自省:本地快模型优先,回退云端(快慢分层)
        verdict = self._grade_llm(prompt)
        grade = self._parse_verdict(verdict)
        via = "local" if self.local else "llm"
        self._trace("grade", via=via, verdict=verdict[:20],
                    elapsed_ms=round((time.time() - t0) * 1000))
        return {"grade": grade, "grade_via": via}

    @staticmethod
    def _parse_verdict(raw: str) -> str:
        """从 JSON 或文本提取 verdict(结构化输出 + 文本回退)。"""
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                v = json.loads(m.group(0)).get("verdict", "")
                if v:
                    return "fail" if "不充分" in v else "pass"
            except Exception:
                pass
        # 回退:纯文本关键词(注意"不充分"含"充分",先判"不充分")
        return "fail" if "不充分" in raw else "pass"

    def _rewrite(self, state: AgentState) -> dict:
        """改写:检索不充分时,把问题改写得更利于检索。"""
        prompt = (
            "用户问题在知识库中检索不到满意结果。请把问题改写得更具体、更利于检索。\n"
            "只输出改写后的问题,不要解释。\n"
            "示例:原始问题=这个怎么用? → 改写=Python requests 库怎么发送 GET 请求?\n\n"
            f"原始问题:{state['query']}\n"
        )
        new_q = self._rewrite_llm(prompt)
        if not new_q or "抱歉" in new_q:
            new_q = state["query"]  # 改写失败则保持原问题,避免卡死
        self._trace("rewrite", old=state["query"][:60], new=new_q[:60])
        return {"query": new_q, "rewrite_count": state.get("rewrite_count", 0) + 1}

    def _generate(self, state: AgentState) -> dict:
        """生成:复用 RAG 生成(reasoning high)。

        整改(低置信度降级):改写次数用尽仍判"不充分"时,不再
        无条件强制生成——提示词追加"证据不足须明示、严禁编造"的系统指令,
        让模型 grounded abstention(只答证据支撑的部分,其余明说证据不足)。

        整改(上下文组装与记忆注入对齐架构图):记忆提示文本(画像+
        历史事实)前置注入;随后走三档上下文组装(≤8 直塞/9-15 截断 top8/
        >15 分层压缩),与经典路径同口径。
        """
        t0 = time.time()
        contexts = [d["text"] for d in state.get("docs", [])]
        hint = (state.get("memory_hint") or "").strip()
        if hint:
            contexts = [hint] + contexts
        try:
            contexts = self.rag._vs.build_layered_contexts(state["query"], contexts)
        except Exception as e:
            logger.warning("agent 路径三档上下文组装失败,用原文兜底: %s", e)
        low_confidence = state.get("grade") == "fail"
        if low_confidence:
            prompt = self.rag._build_rag_prompt(state["query"], contexts)
            prompt += (
                "\n\n【系统指令】检索到的知识库证据不足以完全支撑回答。"
                "只回答能由上述证据支撑的部分;无法支撑的部分必须明确说明"
                "\"知识库证据不足\",严禁编造事实或装作知道。"
            )
            answer = self.rag._call_llm_with_retry(
                prompt, max_tokens=4096, reasoning="high", fallback=""
            )
            if not answer:
                answer = self.rag._call_llm_with_retry(
                    prompt, max_tokens=2048, reasoning="low",
                    fallback="抱歉，知识库证据不足，暂时无法可靠回答该问题。",
                )
        else:
            answer = self.rag._generate(state["query"], contexts)
        # O2 成本量化:记录输入上下文字符数(≈输入 token)与输出长度(≈输出 token)
        self._trace("generate", answer_len=len(answer or ""),
                    context_chars=sum(len(c) for c in contexts),
                    prompt_version=PROMPT_VERSION,
                    low_confidence=low_confidence,
                    elapsed_ms=round((time.time() - t0) * 1000))
        return {"answer": answer}

    # ---------- 路由 ----------
    def _route_after_grade(self, state: AgentState) -> str:
        if state.get("grade") == "pass":
            return "generate"
        if state.get("rewrite_count", 0) < MAX_REWRITES:
            return "rewrite"
        return "generate"  # 改写次数用尽仍生成(兜底)

    # ---------- 建图 ----------
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("rewrite", self._rewrite)
        g.add_node("generate", self._generate)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade", self._route_after_grade,
            {"generate": "generate", "rewrite": "rewrite"},
        )
        g.add_edge("rewrite", "retrieve")
        g.add_edge("generate", END)
        return g.compile()

    # ---------- 入口 ----------
    def query(self, user_query: str, verbose: bool = False, memory_hint: str = "") -> dict:
        state: AgentState = {
            "query": user_query,
            "original_query": user_query,
            "docs": [],
            "grade": "",
            "rewrite_count": 0,
            "answer": "",
            "memory_hint": memory_hint or "",
        }
        try:
            result = self._graph.invoke(state)
        except Exception as e:
            logger.warning("Agent 循环失败,降级为单次直接检索: %s", e)
            return self._fallback_query(user_query)
        if verbose:
            print(f"[AdaptiveRAG] 问题: {user_query}")
            print(f"  改写次数: {result.get('rewrite_count', 0)}")
            print(f"  最终自省: {result.get('grade', '')}")
            print(f"  检索块数: {len(result.get('docs', []))}")
            print(f"  答案: {(result.get('answer') or '')[:200]}")
        return {
            "answer": result.get("answer", ""),
            "sources": [d["text"] for d in result.get("docs", [])],
            "rewrite_count": result.get("rewrite_count", 0),
            "degraded": False,
        }

    def retrieve_loop(self, user_query: str, memory_hint: str = "") -> dict:
        """只跑检索-自省-改写循环(不生成),返回最终 docs 与改写次数。

        供评测 harness 对比:单次检索 vs agent 循环检索的命中率。
        """
        state: AgentState = {
            "query": user_query,
            "original_query": user_query,
            "docs": [],
            "grade": "",
            "grade_via": "",
            "rewrite_count": 0,
            "answer": "",
            "memory_hint": memory_hint or "",
        }
        state.update(self._retrieve(state))
        for _ in range(MAX_REWRITES + 1):
            state.update(self._grade(state))
            if state["grade"] == "pass":
                break
            if state["rewrite_count"] >= MAX_REWRITES:
                break
            state.update(self._rewrite(state))
            state.update(self._retrieve(state))
        return {
            "docs": state["docs"],
            "rewrite_count": state["rewrite_count"],
            "grade": state["grade"],
            "grade_via": state["grade_via"],
        }

    def _fallback_query(self, user_query: str) -> dict:
        """降级兜底:单次检索 + 生成(经典 RAG),保证永远有答案。"""
        try:
            docs = self.rag._vs.hybrid_search_with_rerank_scored(
                user_query, top_k=cfg.RETRIEVAL_K
            )
            contexts = [t for t, _ in docs]
            answer = self.rag._generate(user_query, contexts)
        except Exception as e:
            logger.error("降级检索也失败: %s", e)
            answer, contexts = "抱歉，服务暂时不可用，请稍后重试。", []
        return {
            "answer": answer,
            "sources": contexts,
            "rewrite_count": 0,
            "degraded": True,
        }


# 向后兼容旧名(历史称呼 AgenticRAG,新代码请用 AdaptiveRAG)
AgenticRAG = AdaptiveRAG
