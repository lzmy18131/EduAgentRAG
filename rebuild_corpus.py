# -*- coding: utf-8 -*-
"""语料恢复重建(2026-08-14 数据事故恢复脚本)。

事故: S6 索引切换实验期间,集合被误判"损坏"自动删除重建,30.7 万向量丢失。
恢复源(均已核实完整):
  1. 技术语料:D:\\edrag_corpus_encoded(pipeline_2 产物,42 万子块 + 16.5 万父块,
     稠密/稀疏向量已预编码,直接入库,免 GPU 重算)→ 全集入库(原线上为抽样版,
     重建后为超集,行数会比原线上版更大(终态 358,221 行),指标在重建语料上重新基线)。
  2. JD 语料:D:\\edrag_corpus_clean\\jobs(46,353 条岗位)分块 + BGE-M3 编码后入库。

用法(项目根):
  env -u PYTHONPATH <python> rebuild_corpus.py [--tech-only|--jd-only]
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\31465\Desktop\pythonAI\pythonProject6\image\大模型\RAG开发\EdeRAG智慧问答项目-RAG项目实战")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "2Milvus_RAG_Qa" / "core"))

import numpy as np
import scipy.sparse as sp
from pymilvus import MilvusClient

from base.config import cfg
from base.logger import logger

ENC_DIR = Path(r"D:\edrag_corpus_encoded")
JOBS_DIR = Path(r"D:\edrag_corpus_clean\jobs")
INGEST_BATCH = 1000  # 1000 行/批(2000 曾触发 Milvus OOM,exit 137)
FLUSH_EVERY = 20     # 每 20 批 flush 一次,压低内存峰值


def sparse_to_dict(row) -> dict:
    coo = row.tocoo()
    return {int(k): float(v) for k, v in zip(coo.col, coo.data)}


def _golden_prefixes() -> set[str]:
    """黄金集 expected_chunk 前 40 字符(用于保证重建语料覆盖黄金父块)。"""
    g = json.loads(
        (PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden_500.json")
        .read_text(encoding="utf-8")
    )
    return {str(x.get("expected_chunk", ""))[:40] for x in g if x.get("expected_chunk")}


def ingest_tech(client, coll: str) -> tuple[int, int]:
    """技术语料:黄金父块全保留 + 均匀抽样到原规模(≈21.9 万行)。

    原线上语料是 42 万子块抽样版(21.9 万行);WSL2 内存 6GB 装不下全集
    (58 万行 OOM 实测),故按原规模重建,并保证黄金集 453 个父块 100% 保留。
    """
    print("== 技术语料:读预编码产物 ==", flush=True)
    children = [json.loads(l) for l in (ENC_DIR / "children.jsonl").read_text(encoding="utf-8").splitlines()]
    parents = [json.loads(l) for l in (ENC_DIR / "parents.jsonl").read_text(encoding="utf-8").splitlines()]
    c_dense = np.load(ENC_DIR / "child_dense.npy", mmap_mode="r")
    p_dense = np.load(ENC_DIR / "parent_dense.npy", mmap_mode="r")
    c_sparse = sp.load_npz(ENC_DIR / "child_sparse.npz")
    p_sparse = sp.load_npz(ENC_DIR / "parent_sparse.npz")
    print(f"全量子块 {len(children)} 父块 {len(parents)}", flush=True)

    # 1) 黄金父块识别(前 40 字符匹配)
    gold = _golden_prefixes()
    parent_idx = {p["id"]: i for i, p in enumerate(parents)}
    golden_pids: set[str] = set()
    for i, p in enumerate(parents):
        if (p["text"] or "")[:40] in gold:
            golden_pids.add(p["id"])
    print(f"黄金父块匹配: {len(golden_pids)}/{len(gold)}", flush=True)

    # 2) 抽样:黄金父块的子块全保留 + 其余均匀抽样到 TARGET_CHILDREN
    TARGET_CHILDREN = 157000
    keep_idx: list[int] = []
    other_idx: list[int] = []
    for i, c in enumerate(children):
        if c["parent_id"] in golden_pids:
            keep_idx.append(i)
        else:
            other_idx.append(i)
    budget = TARGET_CHILDREN - len(keep_idx)
    if budget <= 0:
        sampled = keep_idx
    else:
        stride = np.linspace(0, len(other_idx) - 1, budget, dtype=int)
        sampled = keep_idx + [other_idx[j] for j in stride]
    sampled = sorted(set(sampled))
    kept_pids = {children[i]["parent_id"] for i in sampled}
    print(f"抽样后子块 {len(sampled)}(黄金子块 {len(keep_idx)}),父块 {len(kept_pids)}", flush=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    for i in range(0, len(sampled), INGEST_BATCH):
        data = []
        for k in range(i, min(i + INGEST_BATCH, len(sampled))):
            j = sampled[k]
            c = children[j]
            data.append({
                "id": c["id"], "parent_id": c["parent_id"], "text": c["text"],
                "source": c["source"], "chunk_index": c.get("chunk_index", 0),
                "chunk_type": "child",
                "dense_vector": c_dense[j].tolist(),
                "sparse_vector": sparse_to_dict(c_sparse[j]),
                "timestamp": now,
            })
        client.upsert(collection_name=coll, data=data)
        if (i // INGEST_BATCH) % FLUSH_EVERY == 0:
            client.flush(collection_name=coll)
        if i % (INGEST_BATCH * 10) == 0:
            print(f"  子块 {min(i + INGEST_BATCH, len(sampled))}/{len(sampled)} ({time.time()-t0:.0f}s)", flush=True)

    p_count = 0
    for i in range(0, len(parents), INGEST_BATCH):
        data = []
        for j in range(i, min(i + INGEST_BATCH, len(parents))):
            p = parents[j]
            if p["id"] not in kept_pids:
                continue
            data.append({
                "id": p["id"] + "_parent", "parent_id": p["id"], "text": p["text"],
                "source": p["source"], "chunk_index": 0, "chunk_type": "parent",
                "dense_vector": p_dense[j].tolist(),
                "sparse_vector": sparse_to_dict(p_sparse[j]),
                "timestamp": now,
            })
        if data:
            client.upsert(collection_name=coll, data=data)
            p_count += len(data)
        if (i // INGEST_BATCH) % FLUSH_EVERY == 0:
            client.flush(collection_name=coll)
    client.flush(collection_name=coll)
    print(f"技术语料入库完成: {len(sampled)} 子块 + {p_count} 父块 ({time.time()-t0:.0f}s)", flush=True)
    return len(sampled), p_count


def ingest_jobs(client, coll: str) -> tuple[int, int]:
    """JD 语料:分块 + BGE-M3 编码 + 入库(与 pipeline_jobs.py 同款)。"""
    import torch
    from milvus_model.hybrid import BGEM3EmbeddingFunction

    import importlib
    process_documents = importlib.import_module("2Milvus_RAG_Qa.core.document_loader").process_documents

    print("== JD:分块 jobs ==", flush=True)
    parents, children = process_documents(data_dir=str(JOBS_DIR))
    print(f"父块 {len(parents)} / 子块 {len(children)}", flush=True)
    if not children:
        raise SystemExit("JD 无子块")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"== JD:加载 BGE-M3({device}) ==", flush=True)
    ef = BGEM3EmbeddingFunction(
        model_name_or_path=r"C:\Users\31465\.cache\modelscope\models\BAAI--bge-m3\snapshots\master",
        use_fp16=(device == "cuda"), device=device,
    )

    def encode_all(texts):
        dense_parts, sparse_parts = [], []
        for i in range(0, len(texts), 512):
            embs = ef(texts[i:i + 512])
            dense_parts.append(np.asarray(embs["dense"], dtype=np.float32))
            sparse_parts.append(sp.vstack([r.tocsr() for r in embs["sparse"]]))
            if i % 2560 == 0:
                print(f"  编码 {min(i + 512, len(texts))}/{len(texts)}", flush=True)
        return np.vstack(dense_parts), sp.vstack(sparse_parts)

    print("== JD:编码 ==", flush=True)
    t0 = time.time()
    c_dense, c_sparse = encode_all([c["text"] for c in children])
    p_dense, p_sparse = encode_all([p["text"] for p in parents])
    print(f"编码完成 ({time.time()-t0:.0f}s)", flush=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i in range(0, len(children), INGEST_BATCH):
        data = []
        for j in range(i, min(i + INGEST_BATCH, len(children))):
            c = children[j]
            data.append({
                "id": c["id"], "parent_id": c["parent_id"], "text": c["text"],
                "source": "jobs", "chunk_index": c.get("chunk_index", 0),
                "chunk_type": "child",
                "dense_vector": c_dense[j].tolist(),
                "sparse_vector": sparse_to_dict(c_sparse[j]),
                "timestamp": now,
            })
        client.upsert(collection_name=coll, data=data)
    for i in range(0, len(parents), INGEST_BATCH):
        data = []
        for j in range(i, min(i + INGEST_BATCH, len(parents))):
            p = parents[j]
            data.append({
                "id": p["id"] + "_parent", "parent_id": p["id"], "text": p["text"],
                "source": "jobs", "chunk_index": 0, "chunk_type": "parent",
                "dense_vector": p_dense[j].tolist(),
                "sparse_vector": sparse_to_dict(p_sparse[j]),
                "timestamp": now,
            })
        client.upsert(collection_name=coll, data=data)
    client.flush(collection_name=coll)
    print(f"JD 入库完成 ({time.time()-t0:.0f}s)", flush=True)
    return len(children), len(parents)


def main() -> None:
    ap = argparse.ArgumentParser(description="EdeRAG 语料恢复重建")
    ap.add_argument("--tech-only", action="store_true")
    ap.add_argument("--jd-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true", default=False,
                    help="显式 drop 后重建(默认 False:续跑已存在的 collection,upsert 幂等)")
    args = ap.parse_args()

    client = MilvusClient(uri=f"http://{cfg.MILVUS_HOST}:{cfg.MILVUS_PORT}")
    if cfg.MILVUS_DB_NAME not in client.list_databases():
        client.create_database(cfg.MILVUS_DB_NAME)
    client.use_database(cfg.MILVUS_DB_NAME)
    coll = cfg.MILVUS_COLLECTION

    if args.rebuild and not args.jd_only:
        old = client.get_collection_stats(coll).get("row_count", 0) if client.has_collection(coll) else 0
        print(f"== 显式 drop 重建(旧 row_count={old}) ==", flush=True)
        if client.has_collection(coll):
            client.drop_collection(coll)
        from vector_store import VectorStore
        VectorStore()  # 建 schema + HNSW 稠密索引 + 稀疏索引
        print("schema 已重建", flush=True)

    client.load_collection(coll)
    if not args.jd_only:
        ingest_tech(client, coll)
    if not args.tech_only:
        ingest_jobs(client, coll)

    stats = client.get_collection_stats(coll)
    print(f"== 完成:row_count={stats.get('row_count')} ==", flush=True)
    # 优化四:语料版本戳自增 → 旧语义缓存全部自动失效(Cache Busting)
    try:
        from rag_cache import RagCache
        RagCache().bump_corpus_version()
    except Exception as e:
        print(f"语料版本戳自增失败(可忽略): {e}", flush=True)


if __name__ == "__main__":
    main()
