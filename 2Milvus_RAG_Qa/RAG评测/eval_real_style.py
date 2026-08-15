# -*- coding: utf-8 -*-
"""真实口语风格评测子集报告(整改 B6):与 LLM 生成评测集分开出指标。

技术类条目:向量检索 HitRate@5/MRR(与主 harness 同口径,Wilson CI + bootstrap);
求职类条目:S4 同口径条件满足率(双路 JD 结构化命中判定)。
口径声明:该子集是"人工编写、模拟真实学员口语化问法"的分布外探测,
不是线上真实流量;指标单独报告,不与 453 条合成集混算。

用法(项目根):
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.eval_real_style
"""
import importlib
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_rag_main = importlib.import_module("2Milvus_RAG_Qa.core.rag_main")
_rag_system = importlib.import_module("2Milvus_RAG_Qa.core.rag_system")
_dual = importlib.import_module("2Milvus_RAG_Qa.core.dual_retrieval")
_jd = importlib.import_module("2Milvus_RAG_Qa.core.jd_structured")
_hit = importlib.import_module("2Milvus_RAG_Qa.RAG评测.hit_rate_eval")
_harness = importlib.import_module("2Milvus_RAG_Qa.RAG评测.eval_harness")

_SET = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_real_style_56.json"
_OUT = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_real_style_report.json"


def _jd_satisfies(row: dict, cond: dict) -> bool:
    if cond.get("city") and row.get("city") != cond["city"]:
        return False
    if cond.get("tech") and row.get("tech_direction") != cond["tech"]:
        return False
    if cond.get("salary_min") and (row.get("salary_max") or 0) < cond["salary_min"]:
        return False
    if cond.get("salary_max") and (row.get("salary_min") or 10**9) > cond["salary_max"]:
        return False
    return True


def main() -> None:
    items = json.loads(_SET.read_text(encoding="utf-8"))
    vs = _rag_main.init_knowledge_base()
    rag = _rag_system.RAGSystem(vector_store=vs)

    tech = [x for x in items if x.get("kind") == "tech_qa"]
    job = [x for x in items if x.get("kind") == "job_condition"]

    # ── 技术类:向量检索 HitRate@5/MRR(与主 harness 同口径) ──
    hits = 0
    reciprocals: list[float] = []
    per_query = []
    for x in tech:
        q = str(x.get("question", ""))
        expected = str(x.get("expected_chunk", ""))
        t0 = time.time()
        recalled = vs.hybrid_search_with_rerank(q, top_k=5)
        ms = round((time.time() - t0) * 1000)
        rank = _hit._first_hit_rank(expected, recalled)
        per_query.append({"question": q, "hit": rank is not None, "rank": rank,
                          "latency_ms": ms, "gold_id": x.get("gold_id")})
        if rank is not None:
            hits += 1
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)
    n = len(tech)
    tech_report = {
        "num_queries": n,
        "hits": hits,
        "hit_rate": round(hits / n, 4) if n else 0.0,
        "mrr": round(sum(reciprocals) / n, 4) if n else 0.0,
        "hit_rate_ci95": [round(x, 4) for x in _harness.wilson_ci(hits, n)],
        "mrr_ci95": [round(x, 4) for x in _harness.bootstrap_mrr_ci(reciprocals)],
        "per_query": per_query,
    }

    # ── 求职类:S4 同口径条件满足率(双路:JD 结构化命中) ──
    ok = 0
    job_per = []
    for x in job:
        q = str(x.get("question", ""))
        cond = x.get("conditions") or {}
        rows = _jd.search_jobs(**cond, limit=5)
        sat = any(_jd_satisfies(r, cond) for r in rows)
        ok += sat
        job_per.append({"question": q, "cond": cond, "satisfied": sat,
                        "num_jd_returned": len(rows)})
    m = len(job)
    job_report = {
        "num_queries": m,
        "condition_satisfied": ok,
        "condition_satisfaction_rate": round(ok / m, 4) if m else 0.0,
        "rate_ci95": [round(x, 4) for x in _harness.wilson_ci(ok, m)],
        "per_query": job_per,
    }

    result = {
        "set": "eval_real_style_56.json",
        "design_notes": (
            "56 条人工编写的口语化/碎片化 query(技术 46 + 求职 10),"
            "问题仅依据黄金集标准答案改写、不接触 chunk 文本,expected_chunk 仅作评分标签;"
            "本子集是分布外探测,指标与 453 条 LLM 生成集分开报告。"
        ),
        "tech_qa": tech_report,
        "job_condition": job_report,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("tech_qa", "job_condition")},
                     ensure_ascii=False, indent=2))
    print(json.dumps({"tech_qa": {k: v for k, v in tech_report.items() if k != "per_query"},
                      "job_condition": {k: v for k, v in job_report.items() if k != "per_query"}},
                     ensure_ascii=False, indent=2))
    print(f"\n报告已写入 {_OUT}")


if __name__ == "__main__":
    main()
