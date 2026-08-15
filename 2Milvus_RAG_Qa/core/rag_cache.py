"""RAG 层 Redis 缓存 — 多轮会话历史 + 语义级响应缓存。

提供两类能力：
    1. 会话历史：按 session_id 存取最近 N 轮对话，供指代消解使用（D11）
    2. 语义响应缓存：命中条件 = embedding 余弦相似度 ≥ 阈值 ∧ 意图一致 ∧
       求职硬槽(city/tech/salary)一致 ∧ corpus/prompt/model 版本一致
       （整改：防"近似 query 不同诉求"误命中，如 北京→上海、薪资不同）

Redis 不可用时所有方法优雅降级为 no-op / None，不影响主流程。
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import redis

from base.config import cfg
from base.logger import logger

# 语义缓存相似度阈值与 TTL
SEMANTIC_SIMILARITY_THRESHOLD = 0.95
SEMANTIC_CACHE_TTL = 24 * 3600  # 24h
SESSION_TTL = 24 * 3600         # 会话历史 24h
SESSION_MAX_TURNS = 10          # 每会话最多保留轮数
CORPUS_VERSION_KEY = "rag_corpus_version"  # 优化四:语料版本戳
DOWNVOTE_INVALIDATE = 5         # 优化四:同一问题连续 5 个 👎 → 缓存软删除


class RagCache:
    """RAG 层 Redis 缓存客户端，Redis 不可用时优雅降级。"""

    def __init__(self) -> None:
        try:
            self._conn = redis.Redis(
                host=cfg.REDIS_HOST,
                port=cfg.REDIS_PORT,
                password=cfg.REDIS_PASSWORD,
                db=cfg.REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            self._conn.ping()
            self.available = True
            logger.info("RAG 缓存 Redis 连接成功")
        except Exception as e:
            logger.warning("RAG 缓存 Redis 不可用，缓存/会话功能关闭: %s", e)
            self._conn = None
            self.available = False

    # ──────────────── 会话历史（D11）───────────────

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"rag_session:{session_id}"

    def get_history(self, session_id: str | None) -> list[dict[str, str]]:
        """读取会话历史，无记录或 Redis 不可用返回空列表。"""
        if not session_id or not self._conn:
            return []
        try:
            raw = self._conn.get(self._history_key(session_id))
            if raw is None:
                return []
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("读取会话历史失败: %s", e)
            return []

    def append_history(
        self,
        session_id: str | None,
        user_query: str,
        answer: str,
        max_turns: int = SESSION_MAX_TURNS,
    ) -> None:
        """追加一轮对话（用户问题 + 助手回答），保留最近 max_turns 轮。"""
        if not session_id or not self._conn:
            return
        try:
            history = self.get_history(session_id)
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": answer})
            # 一轮 = 2 条消息，最多保留 max_turns 轮
            history = history[-max_turns * 2:]
            self._conn.setex(
                self._history_key(session_id),
                SESSION_TTL,
                json.dumps(history, ensure_ascii=False),
            )
        except Exception as e:
            logger.warning("写入会话历史失败: %s", e)

    # ──────────────── 语义响应缓存（D12 + 优化四版本戳）───────────────

    def get_corpus_version(self) -> int:
        """语料版本号(优化四):语料重建后自增,旧缓存自动失效。"""
        if not self._conn:
            return 1
        try:
            v = self._conn.get(CORPUS_VERSION_KEY)
            return int(v) if v is not None else 1
        except Exception:
            return 1

    def bump_corpus_version(self) -> int:
        """语料版本自增(rebuild_corpus.py 重建后调用)。"""
        if not self._conn:
            return 1
        try:
            v = self._conn.incr(CORPUS_VERSION_KEY)
            logger.info("语料版本戳已自增 → v%s,旧语义缓存全部失效", v)
            return int(v)
        except Exception as e:
            logger.warning("语料版本自增失败: %s", e)
            return 1

    @staticmethod
    def _sem_key(query: str, version: int) -> str:
        """语义缓存 key:版本戳 + 规范化文本 md5(语料一变,旧 key 全废)。"""
        normalized = "".join(query.split())
        return (
            f"rag_sem:v{version}:"
            f"{hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]}"
        )

    @staticmethod
    def _down_key(query: str) -> str:
        """点踩计数器 key(按规范化问题)。"""
        normalized = "".join(query.split())
        return f"rag_down:{hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]}"

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """两个向量余弦相似度。"""
        denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def get_semantic(
        self,
        query: str,
        query_emb: np.ndarray,
        threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
        intent: str | None = None,
        slots: dict | None = None,
        prompt_version: str | None = None,
        llm_model: str | None = None,
    ) -> dict[str, Any] | None:
        """查询语义缓存:精确 key 快路径 + 最近邻余弦相似度扫描。

        整改(意图一致性校验):仅凭 embedding 相似度会误命中
        "近似 query 但诉求不同"的缓存(如 北京→上海、薪资不同)。
        命中条件 = 相似度 ≥ 阈值 AND 意图一致 AND 求职类硬槽一致
        AND prompt/模型版本一致 AND 语料版本一致(版本戳在 key 中)。

        Args:
            query: 用户问题(用于精确 key 匹配)
            query_emb: query 的 embedding(用于最近邻相似度计算)
            threshold: 最近邻相似度阈值,≥ 此值视为命中
            intent: 本次查询的意图(与缓存条目校验一致性)
            slots: 求职类硬槽 {city/tech/salary_min/salary_max}(同上)

        Returns:
            命中的缓存 payload(含 answer/sources/intent/strategy),未命中返回 None
        """
        if not self._conn:
            return None
        version = self.get_corpus_version()
        slots = slots or {}
        try:
            # 1. 精确 key 快路径(带版本戳)
            raw = self._conn.get(self._sem_key(query, version))
            if raw is not None:
                payload = json.loads(raw)
                if isinstance(payload, dict) and payload.get("answer") and \
                        self._payload_consistent(payload, intent, slots,
                                                 prompt_version, llm_model):
                    logger.info("语义缓存精确命中: %s", query[:40])
                    return payload

            # 2. 最近邻扫描:遍历当前版本的全部语义缓存条目算余弦相似度,
            #    并校验意图/硬槽/版本一致性
            best: dict[str, Any] | None = None
            best_sim = -1.0
            for key in self._conn.scan_iter(match=f"rag_sem:v{version}:*", count=100):
                try:
                    raw = self._conn.get(key)
                    if raw is None:
                        continue
                    payload = json.loads(raw)
                    emb_list = payload.get("embedding")
                    if not isinstance(emb_list, list) or not payload.get("answer"):
                        continue
                    if not self._payload_consistent(payload, intent, slots,
                                                     prompt_version, llm_model):
                        continue
                    sim = self._cosine(query_emb, np.asarray(emb_list, dtype=np.float32))
                    if sim > best_sim:
                        best_sim = sim
                        best = payload
                except Exception:
                    continue
            if best is not None and best_sim >= threshold:
                logger.info("语义缓存最近邻命中: %s (sim=%.4f)", query[:40], best_sim)
                return best
            return None
        except Exception as e:
            logger.warning("语义缓存查询失败: %s", e)
            return None

    @staticmethod
    def _payload_consistent(
        payload: dict[str, Any],
        intent: str | None,
        slots: dict | None,
        prompt_version: str | None = None,
        llm_model: str | None = None,
    ) -> bool:
        """缓存条目与当前查询的一致性校验(意图/硬槽/版本)。

        任一不一致即视为不同诉求或不同版本,不得复用缓存答案:
          - 意图标签严格相等(非仅求职/非求职粗分);
          - 求职类硬槽(city/tech/salary)逐字段相等;
          - prompt 版本与生成模型版本与当前一致。
        """
        slots = slots or {}
        cached_intent = payload.get("intent")
        if intent is not None and cached_intent is not None and cached_intent != intent:
            return False
        cached_slots = payload.get("slots") or {}
        if slots or cached_slots:
            for k in ("city", "tech", "salary_min", "salary_max"):
                if slots.get(k) != cached_slots.get(k):
                    return False
        if prompt_version is not None:
            cached_pv = payload.get("prompt_version")
            if cached_pv and cached_pv != prompt_version:
                return False
        if llm_model is not None:
            cached_model = payload.get("llm_model")
            if cached_model and cached_model != llm_model:
                return False
        return True

    def set_semantic(
        self,
        query: str,
        query_emb: np.ndarray,
        payload: dict[str, Any],
        ttl: int = SEMANTIC_CACHE_TTL,
    ) -> bool:
        """写入语义缓存，带 TTL。"""
        if not self._conn:
            return False
        try:
            record = dict(payload)
            record["embedding"] = np.asarray(query_emb, dtype=np.float32).tolist()
            record["query"] = query
            record["ts"] = time.time()
            record["corpus_version"] = self.get_corpus_version()
            self._conn.setex(
                self._sem_key(query, record["corpus_version"]),
                ttl,
                json.dumps(record, ensure_ascii=False),
            )
            return True
        except Exception as e:
            logger.warning("语义缓存写入失败: %s", e)
            return False

    def note_downvote(self, query: str, threshold: int = DOWNVOTE_INVALIDATE) -> bool:
        """优化四主动失效:同一问题连续点踩达到阈值 → 软删除其语义缓存。

        返回 True 表示本次触发了缓存删除(进入 E4 紧急回灌流程)。
        """
        if not self._conn:
            return False
        try:
            count = self._conn.incr(self._down_key(query))
            if count >= threshold:
                version = self.get_corpus_version()
                self._conn.delete(self._sem_key(query, version))
                self._conn.delete(self._down_key(query))
                logger.warning("问题连续 %d 个 👎,语义缓存已软删除: %s", count, query[:40])
                return True
            return False
        except Exception as e:
            logger.warning("点踩缓存失效失败: %s", e)
            return False
