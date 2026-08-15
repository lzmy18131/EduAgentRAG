"""检索侧评测 — HitRate@K 与 MRR。

加载 eval_golden_500.json，对每条 question 跑 VectorStore.hybrid_search_with_rerank，
统计期望 chunk 是否出现在召回父块中（文本包含判定，取前 40 字符）。

运行：python -m 2Milvus_RAG_Qa.RAG评测.hit_rate_eval --top-k 5 --dense-weight 0.7 --sparse-weight 0.3
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from base.config import cfg
from base.logger import logger

# 数字开头包需 importlib 导入
_vector_store_module = importlib.import_module("2Milvus_RAG_Qa.core.vector_store")
VectorStore = _vector_store_module.VectorStore

_DEFAULT_GOLDEN = str(
    _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden_500.json"
)
_DEFAULT_REPORT = str(
    _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_report.json"
)


def _is_hit(expected_chunk: str, recalled: list[str]) -> bool:
    """期望 chunk 前 40 字符是否出现在任一召回父块中（文本包含）。"""
    prefix = (expected_chunk or "")[:40]
    if not prefix:
        return False
    return any(prefix in (text or "") for text in recalled)


def _first_hit_rank(expected_chunk: str, recalled: list[str]) -> int | None:
    """返回首个命中召回的 1 基序号，未命中返回 None。"""
    prefix = (expected_chunk or "")[:40]
    if not prefix:
        return None
    for idx, text in enumerate(recalled):
        if prefix in (text or ""):
            return idx + 1
    return None


def run_eval(golden: list[dict], top_k: int) -> dict:
    """对金标逐条检索，返回命中率与 MRR(含逐 query 结果,供配对检验)。"""
    vs = VectorStore()
    hits = 0
    reciprocals: list[float] = []
    per_query: list[dict] = []
    for item in golden:
        question = str(item.get("question", ""))
        expected = str(item.get("expected_chunk", ""))
        recalled = vs.hybrid_search_with_rerank(question, top_k=top_k)
        rank = _first_hit_rank(expected, recalled)
        per_query.append({
            "question": question,
            "hit": rank is not None,
            "rank": rank,
        })
        if rank is not None:
            hits += 1
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)

    total = len(golden)
    return {
        "num_queries": total,
        "hits": hits,
        "hit_rate": hits / total if total else 0.0,
        "mrr": sum(reciprocals) / total if total else 0.0,
        # E5 统计严谨:逐条倒数排名,供 harness 计算 MRR bootstrap 置信区间
        "reciprocals": reciprocals,
        # 整改:逐 query 命中对照,供 McNemar 配对检验使用
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索命中率评测")
    parser.add_argument("--golden", type=str, default=_DEFAULT_GOLDEN, help="金标 JSON 路径")
    parser.add_argument("--top-k", type=int, default=cfg.RETRIEVAL_K, help="召回数量 K")
    parser.add_argument("--dense-weight", type=float, default=None, help="覆盖稠密权重")
    parser.add_argument("--sparse-weight", type=float, default=None, help="覆盖稀疏权重")
    parser.add_argument("--report", type=str, default=_DEFAULT_REPORT, help="报告输出路径")
    args = parser.parse_args()

    # 覆盖 cfg，供网格搜索使用
    if args.dense_weight is not None:
        cfg.DENSE_WEIGHT = args.dense_weight
    if args.sparse_weight is not None:
        cfg.SPARSE_WEIGHT = args.sparse_weight

    with open(args.golden, encoding="utf-8") as f:
        golden = json.load(f)
    logger.info("加载金标 %d 条", len(golden))

    result = run_eval(golden, args.top_k)
    result.update({
        "top_k": args.top_k,
        "dense_weight": cfg.DENSE_WEIGHT,
        "sparse_weight": cfg.SPARSE_WEIGHT,
    })

    print("=" * 50)
    print("  检索评测报告 (HitRate@K / MRR)")
    print("=" * 50)
    print(f"  查询数      : {result['num_queries']}")
    print(f"  top_k       : {result['top_k']}")
    print(f"  dense_weight: {result['dense_weight']}")
    print(f"  sparse_weight: {result['sparse_weight']}")
    print(f"  命中数      : {result['hits']}")
    print(f"  HitRate@{result['top_k']}: {result['hit_rate']:.4f}")
    print(f"  MRR         : {result['mrr']:.4f}")
    print("=" * 50)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"报告已写入 {report_path}")


if __name__ == "__main__":
    main()
