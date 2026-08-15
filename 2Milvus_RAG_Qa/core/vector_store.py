"""Milvus 向量存储模块 — 文档入库 + 稠密稀疏混合检索 + 父子分块 + 重排序。

使用 BGE-M3 作为嵌入模型，同时产出稠密向量和稀疏向量。
"""

import hashlib
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# 国内用 HuggingFace 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── sys.path 注入，确保项目根可访问 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    WeightedRanker,
)
from pymilvus.milvus_client.index import IndexParams, IndexParam
from sentence_transformers import CrossEncoder

from base.config import cfg
from base.logger import logger

# ── 全局配置 ──
_MILVUS_URI = f"http://{cfg.MILVUS_HOST}:{cfg.MILVUS_PORT}"
_COLLECTION_NAME = cfg.MILVUS_COLLECTION
_DB_NAME = cfg.MILVUS_DB_NAME
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── HNSW 索引参数（9b：IVF_FLAT → HNSW，仅对新建 collection 生效）──
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 200
_HNSW_EF_SEARCH = 128

# ── S6 向量量化(IVF_PQ):默认 HNSW,rebuild_dense_index 可切换并实测对比 ──
_IVF_NLIST = 1024   # 聚类数(≈√N 量级,N=30.8 万)
_IVF_M = 64         # PQ 子空间数(1024 维 % 64 == 0)
_IVF_NBITS = 8      # 子空间量化位数
_IVF_NPROBE = 64    # 查询探测聚类数(召回/延迟权衡,消融实测再定)

# ── MMR 多样性重排参数（E14）──
_MMR_LAMBDA = 0.7

# ── X3 时序衰减(默认关闭)—— 文档时效权重:score × (0.5 + 0.5·exp(-days/90))
# 说明:重建语料全部同一时间戳,衰减无意义;待增量语料带真实 doc 时间戳后置 True
# 启用后需用 harness 做衰减开/关消融再定阈值(说明:实测,有实测依据)
_TIME_DECAY_ENABLED = False
_TIME_DECAY_HALF_LIFE_DAYS = 90


class VectorStore:
    """Milvus 向量存储 — 文档嵌入入库 + 混合检索。

    数据结构（子块/父块双存）：
        子块字段：id, child_id, parent_id, text, dense_vector, sparse_vector,
                  source, chunk_index, timestamp
        父块字段：parent_id, text, source, timestamp
    """

    def __init__(self) -> None:
        # 优先用 ModelScope/HuggingFace 本地缓存,避免重复下载
        _local = os.path.expanduser("~/.cache/modelscope/models/BAAI--bge-m3/snapshots/master")
        _model_path = _local if os.path.isdir(_local) else "BAAI/bge-m3"
        if os.path.isdir(_model_path):
            # 本地模型存在时清掉 HF_ENDPOINT,避免 transformers 误连镜像下载 config
            os.environ.pop("HF_ENDPOINT", None)
        self._ef = BGEM3EmbeddingFunction(
            model_name_or_path=_model_path,
            use_fp16=(_DEVICE == "cuda"),
            device=_DEVICE,
        )
        self._reranker: CrossEncoder | None = None
        self._client: MilvusClient | None = None
        self._init_client()
        self._create_or_load_collection()
        self._dense_index_type = self._get_dense_index_type()

    def _get_dense_index_type(self) -> str:
        """读取当前稠密索引类型(HNSW/IVF_PQ...),查询参数据此自适应。"""
        try:
            for name in self._client.list_indexes(_COLLECTION_NAME):
                info = self._client.describe_index(_COLLECTION_NAME, name)
                if info.get("field_name") == "dense_vector":
                    return str(info.get("index_type", "HNSW"))
        except Exception:
            pass
        return "HNSW"

    def rebuild_dense_index(self, index_type: str = "IVF_PQ", **params) -> dict:
        """S6:切换稠密索引类型(仅重建索引层,数据不动,可回滚)。

        Args:
            index_type: "HNSW" 或 "IVF_PQ"
            params:     索引超参(M/efConstruction 或 nlist/m/nbits)

        Returns:
            新索引描述 dict
        """
        if self._client.has_collection(_COLLECTION_NAME) and \
                "dense_idx" in self._client.list_indexes(_COLLECTION_NAME):
            # Milvus 要求先释放再重建索引,完成后重新加载
            self._client.release_collection(_COLLECTION_NAME)
            self._client.drop_index(_COLLECTION_NAME, "dense_idx")
        if index_type == "HNSW":
            p = {
                "M": params.get("M", _HNSW_M),
                "efConstruction": params.get("efConstruction", _HNSW_EF_CONSTRUCTION),
            }
        elif index_type == "IVF_PQ":
            p = {
                "nlist": params.get("nlist", _IVF_NLIST),
                "m": params.get("m", _IVF_M),
                "nbits": params.get("nbits", _IVF_NBITS),
            }
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")
        index_params = IndexParams()
        index_params.append(IndexParam(
            index_name="dense_idx",
            field_name="dense_vector",
            index_type=index_type,
            metric_type="IP",
            params=p,
        ))
        self._client.create_index(
            collection_name=_COLLECTION_NAME,
            index_params=index_params,
        )
        self._client.load_collection(_COLLECTION_NAME)
        self._dense_index_type = index_type
        logger.info("稠密索引切换完成: %s %s", index_type, p)
        return self._client.describe_index(_COLLECTION_NAME, "dense_idx")

    def _dense_search_params(self) -> dict:
        """按当前索引类型返回稠密检索查询参数(HNSW→ef / IVF→nprobe)。"""
        if self._dense_index_type.startswith("IVF"):
            return {"metric_type": "IP", "params": {"nprobe": _IVF_NPROBE}}
        return {"metric_type": "IP", "params": {"ef": _HNSW_EF_SEARCH}}

    # ──────────────── 初始化 ────────────────

    def _init_client(self) -> None:
        """创建 Milvus 客户端，确保数据库存在。"""
        self._client = MilvusClient(uri=_MILVUS_URI)
        if _DB_NAME not in self._client.list_databases():
            self._client.create_database(_DB_NAME)
        self._client.use_database(_DB_NAME)
        logger.info("Milvus 客户端初始化完成，数据库=%s", _DB_NAME)

    def _create_or_load_collection(self) -> None:
        """创建 Collection(含稠密+稀疏索引),如已存在则加载。

        安全约定(2026-08-14 数据事故后):加载失败**绝不**自动删除重建——
        集合里的数据是唯一资产,删库只能由显式 clear()/rebuild 触发。
        """
        if self._client.has_collection(_COLLECTION_NAME):
            self._client.load_collection(_COLLECTION_NAME)
            logger.info("Collection [%s] 已存在,加载成功", _COLLECTION_NAME)
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_index", DataType.INT32)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)  # "child" or "parent"
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("timestamp", DataType.VARCHAR, max_length=32)

        index_params = IndexParams()
        index_params.append(IndexParam(
            index_name="dense_idx",
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="IP",
            params={"M": _HNSW_M, "efConstruction": _HNSW_EF_CONSTRUCTION},
        ))
        index_params.append(IndexParam(
            index_name="sparse_idx",
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        ))

        self._client.create_collection(
            collection_name=_COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        self._client.load_collection(_COLLECTION_NAME)
        logger.info("Collection [%s] 创建并加载成功", _COLLECTION_NAME)

    def _init_reranker(self) -> None:
        """延迟初始化 CrossEncoder（首次混检时加载）。

        优先用 ModelScope 本地缓存，避免直连 HuggingFace 超时（国内网络）。
        """
        if self._reranker is None:
            _local = os.path.expanduser(
                "~/.cache/modelscope/models/BAAI--bge-reranker-large/snapshots/master"
            )
            _model_path = _local if os.path.isdir(_local) else "BAAI/bge-reranker-large"
            self._reranker = CrossEncoder(
                _model_path,
                device=_DEVICE,
            )
            logger.info("CrossEncoder 重排序模型加载完成: %s", _model_path)

    # ──────────────── 文档入库 ────────────────

    @staticmethod
    def _sparse_to_dict(sparse_row) -> dict:
        """将 scipy 稀疏行转为 pymilvus 需要的 {token_id: weight} dict。

        pymilvus 3.x 的 SPARSE_FLOAT_VECTOR 不接受 scipy 对象，必须转 dict。
        """
        coo = sparse_row.tocoo()
        return {int(k): float(v) for k, v in zip(coo.col, coo.data)}

    def add_documents(
        self,
        parent_chunks: list[dict[str, Any]],
        child_chunks: list[dict[str, Any]],
    ) -> int:
        """批量写入父块和子块。

        Args:
            parent_chunks: [{"id":…, "text":…, "source":…}]  父块列表
            child_chunks:  [{"id":…, "text":…, "source":…, "parent_id":…, "chunk_index":…}]  子块列表

        Returns:
            写入的总记录数
        """
        if not child_chunks:
            return 0

        all_chunks = child_chunks + parent_chunks
        child_texts = [c["text"] for c in child_chunks]
        parent_texts = [c["text"] for c in parent_chunks]

        # BGE-M3 批量向量化
        child_embs = self._ef(child_texts)
        parent_embs = self._ef(parent_texts) if parent_texts else {"dense": [], "sparse": []}

        data: list[dict] = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for i, chunk in enumerate(child_chunks):
            data.append({
                "id": chunk["id"],
                "parent_id": chunk["parent_id"],
                "text": chunk["text"],
                "source": chunk.get("source", ""),
                "chunk_index": chunk.get("chunk_index", i),
                "chunk_type": "child",
                "dense_vector": child_embs["dense"][i].tolist(),
                "sparse_vector": self._sparse_to_dict(child_embs["sparse"][i]),
                "timestamp": now,
            })

        for i, chunk in enumerate(parent_chunks):
            data.append({
                "id": chunk["id"] + "_parent",
                "parent_id": chunk["id"],
                "text": chunk["text"],
                "source": chunk.get("source", ""),
                "chunk_index": 0,
                "chunk_type": "parent",
                "dense_vector": parent_embs["dense"][i].tolist(),
                "sparse_vector": self._sparse_to_dict(parent_embs["sparse"][i]),
                "timestamp": now,
            })

        self._client.upsert(collection_name=_COLLECTION_NAME, data=data)
        # flush 立即落盘，否则 get_collection_stats 的 row_count 不会更新
        self._client.flush(collection_name=_COLLECTION_NAME)
        logger.info("入库完成：%d 子块 + %d 父块", len(child_chunks), len(parent_chunks))
        return len(data)

    # ──────────────── 混合检索 ────────────────

    @staticmethod
    def _build_filter_expr(source_filter: str | None) -> str:
        """构建 Milvus 过滤表达式：限定子块 + 可选按 source 过滤。

        Args:
            source_filter: 文档来源（如仓库名），None 表示不过滤

        Returns:
            Milvus 过滤表达式字符串
        """
        filter_expr = 'chunk_type == "child"'
        if source_filter:
            escaped_source = source_filter.replace("\\", "\\\\").replace('"', '\\"')
            filter_expr += f' and source == "{escaped_source}"'
        return filter_expr

    def hybrid_search_with_rerank(
        self,
        query: str,
        top_k: int = 3,
        source_filter: str | None = None,
        skip_rerank: bool = False,
    ) -> list[str]:
        """稠密+稀疏混合检索 → 子块召回 → 父块映射去重 → CrossEncoder 重排序。

        与 hybrid_search_with_rerank_scored 等价，但仅返回文本列表（兼容旧接口）。

        Args:
            query: 用户查询（改写后）
            top_k: 最终返回的上下文数量
            source_filter: 可选，按文档来源过滤
            skip_rerank: True 时跳过 CrossEncoder 精排，直接按 RRF 粗排顺序返回父块前 top_k
                （供消融实验对比精排收益；默认 False，行为与历史版本完全一致）

        Returns:
            重排后的 top_k 文档正文列表
        """
        scored = self.hybrid_search_with_rerank_scored(
            query, top_k=top_k, source_filter=source_filter, skip_rerank=skip_rerank
        )
        return [text for text, _ in scored]

    def hybrid_search_with_rerank_scored(
        self,
        query: str,
        top_k: int = 3,
        source_filter: str | None = None,
        skip_rerank: bool = False,
    ) -> list[tuple[str, float]]:
        """稠密+稀疏混合检索 → 父块去重 → CrossEncoder 精排 → MMR 多样性重排。

        Args:
            query: 用户查询（改写后）
            top_k: 最终返回的上下文数量
            source_filter: 可选，按文档来源过滤
            skip_rerank: True 时跳过 CrossEncoder 精排，直接按 RRF 粗排顺序返回

        Returns:
            [(文档正文, 精排分数), ...] 列表，分数降序（MMR 首元素即精排 top1）
        """
        # 1. BGE-M3 查询向量化
        q_embs = self._ef([query])

        # 2. 构建过滤表达式
        filter_expr = self._build_filter_expr(source_filter)

        # 3. 稠密检索请求(查询参数随索引类型自适应:HNSW→ef / IVF_PQ→nprobe)
        dense_req = AnnSearchRequest(
            data=[q_embs["dense"][0].tolist()],
            anns_field="dense_vector",
            param=self._dense_search_params(),
            limit=top_k * cfg.RERANK_POOL_MULTIPLIER,
        )

        # 4. 稀疏检索请求
        sparse_req = AnnSearchRequest(
            data=[self._sparse_to_dict(q_embs["sparse"][0])],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=top_k * cfg.RERANK_POOL_MULTIPLIER,
        )

        # 5. 混合检索:RRF 融合(不依赖分数尺度,实测 WeightedRanker 因 dense/sparse 尺度差 2 个数量级导致权重失效)
        if cfg.RANKER_TYPE == "rrf":
            from pymilvus import RRFRanker
            ranker = RRFRanker(k=cfg.RRF_K)
        else:
            ranker = WeightedRanker(cfg.DENSE_WEIGHT, cfg.SPARSE_WEIGHT)
        hybrid_results = self._client.hybrid_search(
            collection_name=_COLLECTION_NAME,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=top_k * cfg.RERANK_POOL_MULTIPLIER // 2,
            output_fields=["parent_id"],
            filter=filter_expr,
        )

        if not hybrid_results or not hybrid_results[0]:
            logger.warning("混合检索无结果: %s", query[:50])
            return []

        # 6. 去重：多个子块映射到同一父块
        seen_parents: dict[str, str] = {}
        candidate_parent_ids: list[str] = []
        for hit in hybrid_results[0]:
            pid = hit["entity"].get("parent_id", "")
            if pid and pid not in seen_parents:
                candidate_parent_ids.append(pid)
                seen_parents[pid] = hit["entity"].get("text", "")

        # 7. 取父块原文；skip_rerank 时按 RRF 粗排顺序直取，跳过精排模型
        parent_texts = self._fetch_parent_texts(candidate_parent_ids)
        if skip_rerank:
            return [(text, 0.0) for text in parent_texts[:top_k]]
        if len(parent_texts) < 2:
            return [(text, 1.0) for text in parent_texts[:top_k]]
        self._init_reranker()
        pairs = [[query, text] for text in parent_texts]
        scores = self._reranker.predict(pairs)

        # X3 时序衰减(默认关闭):精排分 × 时效权重,旧文档自然沉底
        if _TIME_DECAY_ENABLED:
            ts_map = self._fetch_parent_timestamps(candidate_parent_ids)
            scores = [
                float(s) * self.time_decay_weight(
                    ts_map.get(pid, ""), half_life_days=_TIME_DECAY_HALF_LIFE_DAYS)
                for s, (_, pid) in zip(scores, zip(parent_texts, candidate_parent_ids))
            ]

        ranked = sorted(zip(parent_texts, scores), key=lambda x: x[1], reverse=True)

        # 8. MMR 多样性重排（E14：防止 top-k 同文档重复）
        if len(ranked) > top_k:
            ranked = self._mmr_rerank(ranked, top_k)

        result = [(text, float(score)) for text, score in ranked[:top_k]]
        logger.info("混合检索完成：召回%d个父块 → 重排返回%d", len(candidate_parent_ids), len(result))
        return result

    @staticmethod
    def _mmr_select(
        scores: list[float], sim_matrix: np.ndarray, lambda_: float, top_k: int
    ) -> list[int]:
        """MMR 多样性选择（纯函数，便于单测）。

        Args:
            scores: (N,) 相关性分数（如精排分）
            sim_matrix: (N, N) 两两余弦相似度矩阵
            lambda_: 相关性权重（0~1），越大越偏相关性
            top_k: 需选出的数量

        Returns:
            被选中项在原始列表中的索引，按选择顺序排列
        """
        n = len(scores)
        remaining = list(range(n))
        selected: list[int] = []
        while remaining and len(selected) < top_k:
            best_i = remaining[0]
            best_mmr = -float("inf")
            for i in remaining:
                if not selected:
                    mmr = scores[i]
                else:
                    max_sim = max(sim_matrix[i][j] for j in selected)
                    mmr = lambda_ * scores[i] - (1 - lambda_) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_i = i
            selected.append(best_i)
            remaining.remove(best_i)
        return selected

    def _mmr_rerank(
        self, ranked: list[tuple[str, float]], top_k: int
    ) -> list[tuple[str, float]]:
        """对精排结果做 MMR 多样性重排。

        仅对精排前 2*top_k 个候选做 embedding，控制计算成本。

        Args:
            ranked: 精排后的 [(text, score)] 列表，score 降序
            top_k: 需选出的数量

        Returns:
            MMR 重排后的 [(text, score)] 列表
        """
        pool = ranked[: top_k * 2]
        if len(pool) <= top_k:
            return pool
        texts = [text for text, _ in pool]
        embs = np.asarray(self._ef(texts)["dense"], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        embs = embs / norms
        sim_matrix = embs @ embs.T
        scores = [float(score) for _, score in pool]
        indices = self._mmr_select(scores, sim_matrix, _MMR_LAMBDA, top_k)
        return [pool[i] for i in indices]

    # ──────────────── 上下文压缩（D10）───────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """按中文/英文句末标点及换行切分句子。"""
        parts = re.split(r"(?<=[。！？!?；;\n])", text)
        return [p.strip() for p in parts if p.strip()]

    def compress_contexts(
        self, query: str, contexts: list[str], max_chars: int | None = None
    ) -> list[str]:
        """句子级相关度过滤：压缩上下文到约 max_chars 字符（D10）。

        对每个上下文的句子做 embedding，与 query 计算余弦相似度，按相关度
        贪心选取句子直到累计字符数达到上限，最后按原上下文顺序还原。

        Args:
            query: 用户查询
            contexts: 精排后的上下文列表
            max_chars: 压缩目标字符上限，None 时取 cfg.CONTEXT_MAX_CHARS

        Returns:
            压缩后的上下文列表
        """
        if not contexts:
            return []
        if not cfg.CONTEXT_COMPRESS_ENABLED:
            return contexts
        max_chars = max_chars or cfg.CONTEXT_MAX_CHARS

        q_emb = np.asarray(self._ef([query])["dense"][0], dtype=np.float32)
        q_norm = float(np.linalg.norm(q_emb)) + 1e-9

        # 分句并记录所属上下文与原始句序
        sentences: list[tuple[int, str]] = []
        for ci, ctx in enumerate(contexts):
            for s in self._split_sentences(ctx):
                sentences.append((ci, s))
        if not sentences:
            return contexts

        texts = [s for _, s in sentences]
        embs = np.asarray(self._ef(texts)["dense"], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        sims = (embs @ q_emb) / (norms.ravel() * q_norm)

        # 按相关度降序贪心选取，直到累计字符数达上限
        order = np.argsort(-sims)
        picked: dict[int, list[tuple[int, str]]] = {}
        total = 0
        for idx in order.tolist():
            ci, s = sentences[idx]
            if total + len(s) > max_chars and picked:
                break
            picked.setdefault(ci, []).append((idx, s))
            total += len(s)

        # 按原上下文顺序 + 原句序还原
        compressed: list[str] = []
        for ci in range(len(contexts)):
            if ci in picked:
                parts = [s for _, s in sorted(picked[ci])]
                if parts:
                    compressed.append("".join(parts))
        return compressed if compressed else contexts[:1]

    # 优化五:自适应上下文路由阈值(整改:补全 8-15 文档区间,单位统一为字符)
    # 全部预算以"字符"为单位(cfg.CONTEXT_MAX_CHARS,中文约 1.5 字符/token,
    # 12k 字符 ≈ 8k token);不再混用"字/12k/2万/2.4万"多口径。
    FULL_PASSTHROUGH_DOCS = 8    # 文档 ≤8 且总字符 ≤ 预算 → 全文直塞不压缩
    TRUNCATE_KEEP_DOCS = 8       # 9-15 文档且 ≤2.4 万字符 → 只保留 top8 原文
    FORCE_COMPRESS_DOCS = 15     # 文档 >15 或总字符 >2.4 万 → 强制分层压缩
    FORCE_COMPRESS_CHARS = 24000

    def build_layered_contexts(
        self,
        query: str,
        contexts: list[str],
        top_full: int = 2,
        max_chars: int | None = None,
    ) -> list[str]:
        """自适应上下文组装(整改后完整规则,堵住原 8-15 文档空洞)。

        路由规则(单位:字符):
          · 文档 ≤8 且总字符 ≤ CONTEXT_MAX_CHARS → 关闭压缩,全文透传
            (召回本就不多,压缩只会丢实体关系);
          · 文档 9-15 且总字符 ≤2.4 万 → **只保留精排 top8 原文,其余丢弃**
            (中量级上下文压缩 ROI 低,截断保留最高信号即可);
          · 文档 >15 或总字符 >2.4 万 → 分层压缩(L1 top_full 原文 +
            L2 句子级相关度压缩,总预算 CONTEXT_MAX_CHARS)。
        动机:压缩不是免费午餐——短上下文压缩损失大、收益小;
        只有上下文真正"长"时压缩才有正收益(压缩的 ROI 阈值)。
        """
        if not contexts:
            return []
        max_chars = max_chars or cfg.CONTEXT_MAX_CHARS
        total_chars = sum(len(c) for c in contexts)

        # 第一档:小上下文全文透传(总字符不超过预算,不会超窗)
        if len(contexts) <= self.FULL_PASSTHROUGH_DOCS and \
                total_chars <= max_chars:
            return contexts

        # 第二档(整改新增):9-15 文档且不超 2.4 万 → 截断保留 top8
        if len(contexts) <= self.FORCE_COMPRESS_DOCS and \
                total_chars <= self.FORCE_COMPRESS_CHARS:
            kept = contexts[:self.TRUNCATE_KEEP_DOCS]
            if sum(len(c) for c in kept) <= max_chars:
                return kept
            return self.compress_contexts(query, kept, max_chars)

        # 第三档:真正长上下文 → 分层压缩
        if len(contexts) <= top_full:
            return contexts if total_chars <= max_chars else \
                self.compress_contexts(query, contexts, max_chars)
        l1 = contexts[:top_full]
        budget = max_chars - sum(len(c) for c in l1)
        if budget <= 0:
            return l1
        l2 = self.compress_contexts(query, contexts[top_full:], max_chars=budget)
        return l1 + l2

    def _fetch_parent_texts(self, parent_ids: list[str]) -> list[str]:
        """根据 parent_id 批量获取父块文本。"""
        if not parent_ids:
            return []

        id_list = '", "'.join(parent_ids)
        results = self._client.query(
            collection_name=_COLLECTION_NAME,
            filter=f'chunk_type == "parent" and parent_id in ["{id_list}"]',
            output_fields=["text", "parent_id"],
            limit=len(parent_ids),
        )
        return [r.get("text", "") for r in results]

    def _fetch_parent_timestamps(self, parent_ids: list[str]) -> dict[str, str]:
        """X3:批量取父块 timestamp 字段(时序衰减用)。"""
        if not parent_ids:
            return {}
        id_list = '", "'.join(parent_ids)
        results = self._client.query(
            collection_name=_COLLECTION_NAME,
            filter=f'chunk_type == "parent" and parent_id in ["{id_list}"]',
            output_fields=["parent_id", "timestamp"],
            limit=len(parent_ids),
        )
        return {r.get("parent_id", ""): r.get("timestamp", "") for r in results}

    @staticmethod
    def time_decay_weight(
        timestamp_str: str, half_life_days: int = 90, now=None
    ) -> float:
        """X3:文档时效权重 0.5 + 0.5·exp(-days/half_life)。

        90 天 → ~0.68;180 天 → ~0.57;渐近 0.5(不归零=旧文档仍可被召回)。
        timestamp_str 为空(未知时间)时返回 1.0(不惩罚)。
        """
        if not timestamp_str:
            return 1.0
        try:
            ts = datetime.strptime(timestamp_str[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return 1.0
        days = max(0.0, ((now or datetime.now()) - ts).total_seconds() / 86400.0)
        return 0.5 + 0.5 * math.exp(-days / max(1, half_life_days))

    # ──────────────── 工具方法 ────────────────

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询为稠密向量（dense），供语义缓存相似度计算使用。"""
        return np.asarray(self._ef([query])["dense"][0], dtype=np.float32)

    @staticmethod
    def make_hash_id(text: str) -> str:
        """用文本内容 MD5 生成唯一 ID。"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    def count_chunks(self) -> int:
        """统计当前 Collection 中的记录数。"""
        try:
            stats = self._client.get_collection_stats(_COLLECTION_NAME)
            return stats.get("row_count", 0)
        except Exception:
            return 0

    def delete_document(self, source_path: str) -> None:
        """删除某个来源文件的所有父子块。"""
        source_id = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]
        escaped = source_id.replace("\\", "\\\\").replace('"', '\\"')
        self._client.delete(
            collection_name=_COLLECTION_NAME,
            filter=f'id like "{escaped}%"',
        )

    def replace_document(
        self,
        source_path: str,
        parent_chunks: list[dict[str, Any]],
        child_chunks: list[dict[str, Any]],
    ) -> int:
        """先 upsert 新块，再删除该文件不再存在的旧 ID。"""
        source_id = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]
        escaped = source_id.replace("\\", "\\\\").replace('"', '\\"')
        written = self.add_documents(parent_chunks, child_chunks)
        current_ids = [chunk["id"] for chunk in child_chunks]
        current_ids.extend(f'{chunk["id"]}_parent' for chunk in parent_chunks)
        quoted_ids = ", ".join(f'"{item_id}"' for item_id in current_ids)
        stale_filter = f'id like "{escaped}%" and id not in [{quoted_ids}]'
        self._client.delete(collection_name=_COLLECTION_NAME, filter=stale_filter)
        return written

    def clear(self) -> None:
        """清空 Collection 并重建。"""
        if self._client.has_collection(_COLLECTION_NAME):
            self._client.drop_collection(_COLLECTION_NAME)
        self._create_or_load_collection()
        logger.info("Collection [%s] 已清空重建", _COLLECTION_NAME)
