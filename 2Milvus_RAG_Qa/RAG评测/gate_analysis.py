# -*- coding: utf-8 -*-
"""级联门控阈值分析(整改):用全量 453 条黄金 query 的门控分数分布做
阈值扫描 + 敏感性分析,替代原先 60 条小样本的单点结论。

回答:
  "0.7 / 0.4 阈值怎么定的?有没有敏感性分析?FP 率多少?"
方法(无需本地模型,只用检索分数与命中标签):
  1. 对 453 条黄金 query 跑检索,记录向量路 top1 重排分 + 命中标签;
  2. 阈值扫描:上界 t ∈ [0.55, 0.85],下界 b ∈ [0.30, 0.50];
     每个 (b, t) 组合计算:
       - 直过率(top1 ≥ t)、直过但未命中率(FP,带 Wilson CI)
       - 直拒率(top1 ≤ b)、直拒但实际命中率(FN,带 Wilson CI)
       - 混沌区占比(即 LLM 仲裁调用率)与其命中占比
  3. 敏感性:对 (0.4, 0.7) 做 ±0.05 抖动,报各指标波动幅度;
  4. 混沌区 0.5B 仲裁准确率另行在 grade_quality.py 评测(已扩展为全量混沌区)。

用法(项目根):
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.gate_analysis
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_rag_main = importlib.import_module("2Milvus_RAG_Qa.core.rag_main")
_rag_system = importlib.import_module("2Milvus_RAG_Qa.core.rag_system")
_harness = importlib.import_module("2Milvus_RAG_Qa.RAG评测.eval_harness")
_hit = importlib.import_module("2Milvus_RAG_Qa.RAG评测.hit_rate_eval")

_DEFAULT_GOLDEN = str(
    _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden_500.json"
)
_OUT = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "gate_analysis.json"


def _rate_wilson(hits: int, n: int) -> dict:
    low, high = _harness.wilson_ci(hits, n)
    return {"hits": hits, "n": n, "rate": round(hits / n, 4) if n else 0.0,
            "ci95": [round(low, 4), round(high, 4)]}


def run_analysis(golden: list[dict]) -> dict:
    vs = _rag_main.init_knowledge_base()
    rag = _rag_system.RAGSystem(vector_store=vs)
    scored_rows = []  # (top1_vector_score, is_hit)
    for item in golden:
        q = str(item.get("question", ""))
        expected = str(item.get("expected_chunk", ""))
        docs = rag._retrieve_scored("直接检索", q)
        top1 = rag._top_vector_score(docs)
        vec_texts = [d["text"] for d in docs if d.get("source") != "jd"]
        is_hit = _hit._first_hit_rank(expected, vec_texts) is not None
        scored_rows.append({"question": q[:40], "top1": round(top1, 4) if top1 is not None else None,
                            "hit": is_hit})
    valid = [r for r in scored_rows if r["top1"] is not None]
    n = len(valid)

    # ── 阈值扫描 ──
    sweep = []
    for t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        for b in [0.30, 0.35, 0.40, 0.45, 0.50]:
            if b >= t:
                continue
            passed = [r for r in valid if r["top1"] >= t]
            failed = [r for r in valid if r["top1"] <= b]
            chaos = [r for r in valid if b < r["top1"] < t]
            fp = sum(1 for r in passed if not r["hit"])          # 直过但未命中
            fn = sum(1 for r in failed if r["hit"])              # 直拒但实际命中
            sweep.append({
                "lower_b": b, "upper_t": t,
                "pass": _rate_wilson(len(passed), n),
                "pass_miss_fp": _rate_wilson(fp, len(passed)),
                "auto_fail": _rate_wilson(len(failed), n),
                "auto_fail_hit_fn": _rate_wilson(fn, len(failed)),
                "chaos": _rate_wilson(len(chaos), n),
                "chaos_hit_share": _rate_wilson(sum(1 for r in chaos if r["hit"]), len(chaos)),
            })

    # ── 敏感性:围绕生产配置 (0.4, 0.7) 抖动 ±0.05 ──
    def _row(b: float, t: float) -> dict:
        passed = [r for r in valid if r["top1"] >= t]
        failed = [r for r in valid if r["top1"] <= b]
        chaos = [r for r in valid if b < r["top1"] < t]
        return {
            "lower_b": b, "upper_t": t,
            "pass_rate": round(len(passed) / n, 4),
            "fp_rate": round(sum(1 for r in passed if not r["hit"]) / max(len(passed), 1), 4),
            "fail_rate": round(len(failed) / n, 4),
            "fn_rate": round(sum(1 for r in failed if r["hit"]) / max(len(failed), 1), 4),
            "chaos_rate": round(len(chaos) / n, 4),
        }

    sensitivity = {
        "production": _row(0.4, 0.7),
        "upper_minus_005": _row(0.4, 0.65),
        "upper_plus_005": _row(0.4, 0.75),
        "lower_minus_005": _row(0.35, 0.7),
        "lower_plus_005": _row(0.45, 0.7),
    }

    result = {
        "num_queries": n,
        "score_distribution": {
            "p10": round(sorted(r["top1"] for r in valid)[int(n * 0.10)], 4),
            "p50": round(sorted(r["top1"] for r in valid)[int(n * 0.50)], 4),
            "p90": round(sorted(r["top1"] for r in valid)[int(n * 0.90)], 4),
            "hit_top1_median": round(sorted(r["top1"] for r in valid if r["hit"])[
                len([r for r in valid if r["hit"]]) // 2], 4) if any(r["hit"] for r in valid) else None,
            "miss_top1_median": round(sorted(r["top1"] for r in valid if not r["hit"])[
                len([r for r in valid if not r["hit"]]) // 2], 4) if any(not r["hit"] for r in valid) else None,
        },
        "threshold_sweep": sweep,
        "sensitivity": sensitivity,
        "notes": (
            "分数口径=向量路 CrossEncoder(bge-reranker-large)重排分 top1,"
            "JD 结构化槽位不参与门控;样本=453 条黄金集(与核心 HitRate 同源);"
            "混沌区 0.5B 仲裁准确率见 grade_quality.json(已扩展为全量混沌区样本)。"
        ),
    }
    _OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="级联门控阈值分析与敏感性")
    parser.add_argument("--golden", default=_DEFAULT_GOLDEN)
    args = parser.parse_args()
    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    result = run_analysis(golden)
    print("=== 分数分布(向量路 top1 重排分) ===")
    print(json.dumps(result["score_distribution"], ensure_ascii=False, indent=2))
    print("=== 敏感性(生产配置 0.4/0.7 抖动 ±0.05) ===")
    print(json.dumps(result["sensitivity"], ensure_ascii=False, indent=2))
    print(f"\n报告已写入 {_OUT}")


if __name__ == "__main__":
    main()
