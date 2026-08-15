"""BM25 检索模块 — 基于 BM25L 算法的稀疏检索核心。

使用方式:
    from retrieval.bm25_search import BM25Search
    searcher = BM25Search(redis_client, mysql_client)
    answer, msg = searcher.search("什么是机器学习？")
"""

import numpy as np
from rank_bm25 import BM25L
from typing import List

from base.logger import logger
from ..cache.redis_client import redis_client
from ..db.mysql_client import mysql_client
from ..utils.process import process_text


class BM25Search:
    """BM25L 检索器 — 对用户问题进行分词后检索最匹配的知识库问题。

    检索流程:
        1. Redis 缓存命中 → 直接返回
        2. jieba 分词 → BM25L 打分 → softmax 归一化
        3. 最高分 ≥ 阈值 → MySQL 查标准答案 → 写入 Redis 缓存
        4. 最高分 < 阈值 → 返回 None，触发外部 RAG 降级
    """

    def __init__(self, redis_client, mysql_client) -> None:
        self._redis = redis_client
        self._mysql = mysql_client

        # 知识库数据
        self.raw_questions: List[str] = []
        self.tokenized_questions: List[List[str]] = []
        self._bm25_model: BM25L | None = None

        self._load_data()

    # ────────────────── 数据加载 ──────────────────

    def _load_data(self) -> None:
        """加载知识库数据：优先 Redis 缓存，否则从 MySQL 读取并写入 Redis。"""
        cache_raw_key = "bm25:raw_questions"
        cache_token_key = "bm25:tokenized_questions"

        # 优先从 Redis 加载
        raw = self._redis.get_data(cache_raw_key) if self._redis.available else None
        tokenized = self._redis.get_data(cache_token_key) if self._redis.available else None

        if raw and tokenized:
            self.raw_questions = raw
            self.tokenized_questions = tokenized
            logger.info("从 Redis 加载 BM25 知识库 (%d 条)", len(self.raw_questions))
        else:
            # Redis 无缓存，从 MySQL 加载
            self.raw_questions = self._mysql.fetch_questions()
            if not self.raw_questions:
                logger.warning("知识库为空，BM25 无法初始化")
                return

            self.tokenized_questions = [process_text(q) for q in self.raw_questions]

            # 写入 Redis 永久缓存
            if self._redis.available:
                self._redis.set_data(cache_raw_key, self.raw_questions)
                self._redis.set_data(cache_token_key, self.tokenized_questions)
            logger.info("从 MySQL 加载 %d 条问题并写入 Redis 缓存", len(self.raw_questions))
        # 初始化 BM25L 模型
        self._bm25_model = BM25L(self.tokenized_questions)
        logger.info("BM25L 检索模型初始化完成")

    # ────────────────── 检索算法 ──────────────────

    @staticmethod
    def softmax(scores: np.ndarray) -> np.ndarray:
        """Softmax 归一化，将 BM25 原始分转为概率分布。

        Args:
            scores: 一维 numpy 数组，每篇文档的 BM25 得分

        Returns:
            归一化后的概率分布，所有值之和为 1
        """
        # 减去最大值防止数值溢出
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)

    def search(self, query: str, threshold: float = 0.85) -> tuple[str | None, str]:
        """检索与用户问题最匹配的知识库答案。

        Args:
            query: 用户输入的问题
            threshold: 置信度阈值 (0~1)，低于此值则不返回答案

        Returns:
            (answer, message) 元组:
                - answer: 匹配成功时返回答案字符串，否则为 None
                - message: 检索状态描述
        """
        # ── 输入校验 ──
        if not isinstance(query, str) or not query.strip():
            logger.warning("非法输入: %s", repr(query))
            return None, "输入问题不能为空"

        if self._bm25_model is None:
            return None, "知识库为空，检索不可用"

        # ── 第 1 步：Redis 问答缓存 ──
        if self._redis.available:
            cached = self._redis.get_answer(query)
            if cached is not None:
                logger.info("Redis 缓存命中: %s", query[:40])
                return cached, "缓存命中"

        # ── 第 2 步：分词 + BM25 打分 ──
        token_query = process_text(query)
        if not token_query:
            return None, "分词结果为空，无法检索"

        scores = self._bm25_model.get_scores(token_query)
        soft_scores = self.softmax(np.array(scores))

        max_idx = int(np.argmax(soft_scores))
        max_score = float(soft_scores[max_idx])
        match_question = self.raw_questions[max_idx]

        # ── 第 3 步：阈值判断 ──
        if max_score < threshold:
            msg = f"最高匹配分 {max_score:.4f} 低于阈值 {threshold}"
            logger.info(msg)
            return None, msg

        # ── 第 4 步：查 MySQL 标准答案 ──
        answer = self._mysql.fetch_answer(match_question)
        if not answer:
            return None, "匹配到问题但数据库无对应答案"

        # ── 第 5 步：写入 Redis 缓存 ──
        if self._redis.available:
            self._redis.set_data(f"rag_answer:{query}", answer)

        logger.info("检索成功 [score=%.4f]: %s → %s", max_score, query[:30], match_question[:30])
        return answer, f"匹配成功，置信度 {max_score:.4f}"
