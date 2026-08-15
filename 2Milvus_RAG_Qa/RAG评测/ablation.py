"""检索侧消融实验 — 对比不同检索策略的 HitRate@K 与 MRR。

以 eval_golden.json 为输入（字段 question/answer/expected_chunk），复用
hit_rate_eval.py 的 _is_hit / _first_hit_rank 判定逻辑
（命中判定 = expected_chunk 前 40 字符出现在召回文本中）。

两种模式：
  1. rerank-ablation：对比「RRF 粗排直接取 top-k」与「RRF + CrossEncoder 精排」两路。
  2. hyde-ablation  ：对比「原始 query 直接检索」与「HyDE 假设答案改写后检索」两路。

运行（项目根）：
  python -m 2Milvus_RAG_Qa.RAG评测.ablation --mode rerank-ablation --top-k 5
  python -m 2Milvus_RAG_Qa.RAG评测.ablation --mode hyde-ablation --top-k 5
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI

from base.config import cfg
from base.logger import logger

# 复用 hit_rate_eval 的命中判定逻辑与 VectorStore（数字开头包需 importlib 导入）
_eval_mod = importlib.import_module("2Milvus_RAG_Qa.RAG评测.hit_rate_eval")
_is_hit = _eval_mod._is_hit
_first_hit_rank = _eval_mod._first_hit_rank
VectorStore = _eval_mod.VectorStore

# 复用 rag_system 的 HyDE 提示词写法
_prompts_mod = importlib.import_module("2Milvus_RAG_Qa.core.prompts")
RAGPrompts = _prompts_mod.RAGPrompts

_DEFAULT_GOLDEN = str(
    _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden.json"
)
_DEFAULT_REPORT = str(
    _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "ablation_report.json"
)

# 控制台对比表的中文标签
_PATH_LABELS: dict[str, dict[str, str]] = {
    "rerank-ablation": {"rrf_coarse": "RRF粗排top-k", "rrf_rerank": "RRF+精排"},
    "hyde-ablation": {"direct": "原始query", "hyde": "HyDE改写"},
}


def _llm_client() -> OpenAI:
    """按 cfg.LLM_* 配置构造 OpenAI 兼容客户端。"""
    return OpenAI(api_key=cfg.LLM_API_KEY, base_url=cfg.LLM_BASE_URL, timeout=120)


def _hyde_rewrite(llm: OpenAI, question: str) -> str:
    """调用 LLM 生成假设答案（HyDE 改写），偶发空返回自动重试 3 次。

    deepseek 实测约 4% 空返回率，故对空返回与调用异常均重试，参数对齐
    generate_golden_set.generate_one（reasoning_effort=high、temperature=0.1）。
    """
    prompt = RAGPrompts.hyde_prompt().format(question=question)
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = llm.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                reasoning_effort="high",
                max_tokens=1024,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
            last_err = ValueError("LLM 返回空内容")
        except Exception as e:  # noqa: BLE001 — 兜底重试，最终失败由调用方处理
            last_err = e
        if attempt < 3:
            time.sleep(2 * attempt)
    raise last_err  # type: ignore[misc]


def _metrics(ranks: list[int | None]) -> dict:
    """由 1 基命中序号列表统计 HitRate@K 与 MRR。"""
    total = len(ranks)
    hits = sum(1 for r in ranks if r is not None)
    mrr = sum(1.0 / r for r in ranks if r is not None)
    return {
        "hits": hits,
        "hit_rate": hits / total if total else 0.0,
        "mrr": mrr / total if total else 0.0,
    }


def run_rerank_ablation(vs: VectorStore, golden: list[dict], top_k: int) -> dict:
    """对比「RRF 粗排直取 top-k」与「RRF + CrossEncoder 精排」两路。"""
    coarse_ranks: list[int | None] = []
    rerank_ranks: list[int | None] = []
    per_item: list[dict] = []

    for item in golden:
        question = str(item.get("question", ""))
        expected = str(item.get("expected_chunk", ""))
        coarse = vs.hybrid_search_with_rerank(question, top_k=top_k, skip_rerank=True)
        reranked = vs.hybrid_search_with_rerank(question, top_k=top_k)

        c_hit = _is_hit(expected, coarse)
        r_hit = _is_hit(expected, reranked)
        c_rank = _first_hit_rank(expected, coarse)
        r_rank = _first_hit_rank(expected, reranked)
        coarse_ranks.append(c_rank)
        rerank_ranks.append(r_rank)

        per_item.append({
            "question": question,
            "expected_chunk": expected,
            "coarse": {"hit": c_hit, "rank": c_rank},
            "rerank": {"hit": r_hit, "rank": r_rank},
            "diff": (c_hit != r_hit) or (c_rank != r_rank),
        })

    return {
        "mode": "rerank-ablation",
        "top_k": top_k,
        "num_queries": len(golden),
        "paths": {
            "rrf_coarse": _metrics(coarse_ranks),
            "rrf_rerank": _metrics(rerank_ranks),
        },
        "per_item": per_item,
    }


def run_hyde_ablation(
    vs: VectorStore, llm: OpenAI, golden: list[dict], top_k: int
) -> dict:
    """对比「原始 query 直接检索」与「HyDE 假设答案改写后检索」两路。"""
    direct_ranks: list[int | None] = []
    hyde_ranks: list[int | None] = []
    per_item: list[dict] = []

    for item in golden:
        question = str(item.get("question", ""))
        expected = str(item.get("expected_chunk", ""))
        direct = vs.hybrid_search_with_rerank(question, top_k=top_k)

        # HyDE 改写失败时回退原始 query，保证单条失败不中断整体评测
        try:
            hyde_q = _hyde_rewrite(llm, question)
            fallback = False
        except Exception as e:  # noqa: BLE001
            logger.warning("HyDE 改写失败，回退原始 query: %s", e)
            hyde_q = question
            fallback = True
        hyde_recalled = vs.hybrid_search_with_rerank(hyde_q, top_k=top_k)

        d_hit = _is_hit(expected, direct)
        h_hit = _is_hit(expected, hyde_recalled)
        d_rank = _first_hit_rank(expected, direct)
        h_rank = _first_hit_rank(expected, hyde_recalled)
        direct_ranks.append(d_rank)
        hyde_ranks.append(h_rank)

        per_item.append({
            "question": question,
            "expected_chunk": expected,
            "hyde_answer": hyde_q,
            "hyde_fallback": fallback,
            "direct": {"hit": d_hit, "rank": d_rank},
            "hyde": {"hit": h_hit, "rank": h_rank},
            "diff": (d_hit != h_hit) or (d_rank != h_rank),
        })

    return {
        "mode": "hyde-ablation",
        "top_k": top_k,
        "num_queries": len(golden),
        "paths": {
            "direct": _metrics(direct_ranks),
            "hyde": _metrics(hyde_ranks),
        },
        "per_item": per_item,
    }


def _print_table(result: dict) -> None:
    """控制台打印两路对比表。"""
    top_k = result["top_k"]
    labels = _PATH_LABELS.get(result["mode"], {})
    print("=" * 60)
    print(f"  消融实验报告 ({result['mode']} @ top_k={top_k})")
    print("=" * 60)
    print(f"  查询数      : {result['num_queries']}")
    print("-" * 60)
    for name, metrics in result["paths"].items():
        label = labels.get(name, name)
        print(
            f"  {label:<12}: HitRate@{top_k}={metrics['hit_rate']:.4f}  "
            f"MRR={metrics['mrr']:.4f}  (命中 {metrics['hits']}/{result['num_queries']})"
        )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索消融实验")
    parser.add_argument(
        "--mode",
        choices=["rerank-ablation", "hyde-ablation"],
        required=True,
        help="消融模式",
    )
    parser.add_argument("--golden", type=str, default=_DEFAULT_GOLDEN, help="金标 JSON 路径")
    parser.add_argument("--top-k", type=int, default=5, help="召回数量 K（默认 5）")
    parser.add_argument("--report", type=str, default=_DEFAULT_REPORT, help="报告输出路径")
    args = parser.parse_args()

    with open(args.golden, encoding="utf-8") as f:
        golden = json.load(f)
    logger.info("加载金标 %d 条", len(golden))

    vs = VectorStore()
    if args.mode == "rerank-ablation":
        result = run_rerank_ablation(vs, golden, args.top_k)
    else:
        llm = _llm_client()
        result = run_hyde_ablation(vs, llm, golden, args.top_k)

    _print_table(result)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"报告已写入 {report_path}")


if __name__ == "__main__":
    main()
