"""FAQ 语义检索模块 — 基于 bge-small-zh-v1.5 的稠密向量语义匹配。

使用方式:
    from retrieval.faq_semantic import FaqSemanticSearch
    searcher = FaqSemanticSearch(redis_client, mysql_client)
    answer, msg = searcher.search("什么是机器学习？")
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from base.logger import logger
from ..utils.process import process_text


class FaqSemanticSearch:
    """FAQ 语义检索器 — 用 bge-small-zh-v1.5 编码问题并做余弦相似度匹配。

    检索流程:
        1. Redis 版本号缓存命中 → 直接返回
        2. encode(query) → matrix @ vec → top-2 分数
        3. top1 >= threshold 且 top1 - top2 >= MARGIN → MySQL 查标准答案 → 写缓存
        4. 否则返回 None，触发外部 RAG 降级
    """

    MATCH_THRESHOLD = 0.90   # 余弦相似度阈值(标定:扩充后342库 0.70~0.95 全 F1=1.0,取 0.90 留难负例缓冲)
    MARGIN = 0.05            # top1-top2 分差，低于视为歧义放行
    HIGH_CONFIDENCE = 0.93   # top1 达到此分数视为高分同义/近似重复，跳过 margin 歧义判定

    def __init__(self, redis_client, mysql_client) -> None:
        self._redis = redis_client
        self._mysql = mysql_client

        # 语义模型与向量矩阵
        self._model: Any | None = None
        self.questions: list[str] = []          # 语料库问题(原问题 + 同义改写)，用于展示与匹配
        self._answer_map: list[str] = []        # 与 questions 等长，记录每条语料回源的原问题(改写无 MySQL 答案)
        self.matrix: np.ndarray | None = None   # (N, 512) 归一化向量矩阵

        self._load_model()
        self._load_data()

    # ────────────────── 模型加载 ──────────────────

    def _load_model(self) -> None:
        """加载 bge-small-zh-v1.5,失败则降级为不可用(不抛异常炸掉启动)。"""
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            from sentence_transformers import SentenceTransformer
            # 优先 ModelScope 本地缓存,避免 hf-mirror 证书/网络问题
            _local = os.path.expanduser(
                "~/.cache/modelscope/models/BAAI--bge-small-zh-v1.5/snapshots/master"
            )
            _model_path = _local if os.path.isdir(_local) else "BAAI/bge-small-zh-v1.5"
            self._model = SentenceTransformer(_model_path, device="cpu")
            logger.info("FAQ 语义模型 bge-small-zh-v1.5 加载成功: %s", _model_path)
        except Exception as e:
            logger.warning("bge-small-zh-v1.5 加载失败,FAQ 语义检索降级不可用: %s", e)
            self._model = None

    # ────────────────── 数据加载 ──────────────────

    def _load_data(self) -> None:
        """从 MySQL 拉取 FAQ 原问题，叠加同义改写扩充语料库并构建向量矩阵。"""
        try:
            originals = self._mysql.fetch_questions()
        except Exception as e:
            logger.warning("拉取 FAQ 问题失败: %s", e)
            originals = []

        if not originals:
            logger.warning("FAQ 知识库为空，语义检索不可用")
            self.questions = []
            self._answer_map = []
            self.matrix = None
            return

        # 加载同义改写(每条原问题 2 条口语化改写)，用于扩充检索库
        paraphrases, para_sources = self._load_paraphrases()

        # 语料库 = 原问题 + 改写；_answer_map 记录每条语料回源的原问题
        self.questions = list(originals) + paraphrases
        self._answer_map = list(originals) + para_sources

        if self._model is None:
            logger.warning("语义模型不可用，跳过 FAQ 向量化")
            self.matrix = None
            return

        try:
            embeddings = self._model.encode(
                self.questions,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )
            self.matrix = np.asarray(embeddings, dtype=np.float32)
            logger.info(
                "FAQ 语义矩阵构建完成：%d 条 × %d 维（含 %d 条同义改写）",
                self.matrix.shape[0], self.matrix.shape[1], len(paraphrases),
            )
        except Exception as e:
            logger.warning("FAQ 向量化失败: %s", e)
            self.matrix = None

    def _load_paraphrases(self) -> tuple[list[str], list[str]]:
        """读取同义改写文件，返回 (改写列表, 对应原问题列表)，两者等长。

        文件格式: [{"question": 原问题, "paraphrases": [改写1, 改写2]}, ...]
        改写由 LLM 按原问题逐条生成，仅用于扩充向量库，本身无 MySQL 标准答案。
        """
        path = Path(__file__).resolve().parent.parent / "data" / "faq_paraphrases.json"
        if not path.is_file():
            logger.warning("同义改写文件不存在，跳过语料扩充: %s", path)
            return [], []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("读取同义改写文件失败: %s", e)
            return [], []

        paraphrases: list[str] = []
        sources: list[str] = []
        for item in data:
            question = item.get("question")
            if not isinstance(question, str) or not question.strip():
                continue
            for p in item.get("paraphrases", []):
                if not isinstance(p, str) or not p.strip():
                    continue
                paraphrases.append(p.strip())
                sources.append(question)
        logger.info("加载同义改写 %d 条", len(paraphrases))
        return paraphrases, sources

    # ────────────────── 检索算法 ──────────────────

    def _encode(self, text: str) -> np.ndarray | None:
        """编码单条查询为归一化向量，失败返回 None。"""
        if self._model is None:
            return None
        try:
            emb = self._model.encode([text], normalize_embeddings=True)
            return np.asarray(emb[0], dtype=np.float32)
        except Exception as e:
            logger.warning("FAQ 查询编码失败: %s", e)
            return None

    def search(self, query: str, threshold: float = 0.90) -> tuple[str | None, str]:
        """检索与用户问题最匹配的 FAQ 标准答案。

        Args:
            query: 用户输入的问题
            threshold: 余弦相似度阈值 (0~1)，低于此值不返回答案

        Returns:
            (answer, message) 元组，answer 为 None 表示未命中需降级 RAG
        """
        # ── 输入校验 ──
        if not isinstance(query, str) or not query.strip():
            logger.warning("非法输入: %s", repr(query))
            return None, "输入问题不能为空"

        # ── 第 1 步：Redis 版本号缓存 ──
        norm_key = "_".join(process_text(query))
        version = self._redis.get_faq_version() if self._redis.available else 1
        if self._redis.available:
            cached = self._redis.get_faq_answer(norm_key, version)
            if cached is not None:
                logger.info("Redis 版本缓存命中: %s", query[:40])
                return cached, "缓存命中"

        # ── 模型/矩阵可用性 ──
        if self._model is None:
            return None, "语义模型加载失败，检索不可用"
        if self.matrix is None or len(self.questions) == 0:
            return None, "知识库为空，检索不可用"

        # ── 第 2 步：编码 + top-2 分数 ──
        vec = self._encode(query)
        if vec is None:
            return None, "查询编码失败，检索不可用"

        scores = self.matrix @ vec  # (N,) 余弦相似度
        top_indices = np.argsort(scores)[::-1][:2]  # 降序取前 2
        top1_idx = int(top_indices[0])
        top1 = float(scores[top1_idx])
        top2 = float(scores[top_indices[1]]) if top_indices.size >= 2 else -1.0

        # ── 第 3 步：阈值与分差判断 ──
        if top1 < threshold:
            msg = f"最高匹配分 {top1:.4f} 低于阈值 {threshold}，未命中降级 RAG"
            logger.info(msg)
            return None, msg

        margin = top1 - top2
        # 扩充库后:top1/top2 若映射到同一原问题(改写变体),分数天然并列,不算语义歧义
        same_source = False
        if self._answer_map and top_indices.size >= 2:
            top2_idx = int(top_indices[1])
            if top2_idx < len(self._answer_map):
                same_source = (self._answer_map[top1_idx] == self._answer_map[top2_idx])
        # top1 高分(≥HIGH_CONFIDENCE)时跳过 margin 判定:高分并列必是同义变体或近似重复原问题
        # (如"什么是过拟合"vs"什么是过拟合?" 分差 0.0111 曾被误伤)
        if margin < self.MARGIN and not same_source and top1 < self.HIGH_CONFIDENCE:
            msg = f"top1/top2 分差 {margin:.4f} 小于 {self.MARGIN}，语义歧义，未命中降级 RAG"
            logger.info(msg)
            return None, msg

        # ── 第 4 步：查 MySQL 标准答案 ──
        match_question = self.questions[top1_idx]
        # 命中同义改写时回源到原问题(改写本身无 MySQL 标准答案)
        if self._answer_map and top1_idx < len(self._answer_map):
            source_question = self._answer_map[top1_idx]
        else:
            source_question = match_question
        answer = self._mysql.fetch_answer(source_question)
        if not answer:
            return None, "匹配到问题但数据库无对应答案"

        # ── 第 5 步：写入版本号缓存 ──
        if self._redis.available:
            self._redis.set_faq_answer(norm_key, version, answer)

        logger.info("检索成功 [score=%.4f]: %s → %s", top1, query[:30], match_question[:30])
        return answer, f"匹配成功，置信度 {top1:.4f}"
