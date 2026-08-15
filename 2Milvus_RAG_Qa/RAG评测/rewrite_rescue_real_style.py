# -*- coding: utf-8 -*-
"""改写机制在"真实口语风格子集"上的救活率评测(整改追问:"改写不是白做吗?")。

背景:453 条合成集上改写救活率 = 0/45(agent_compare.json)——合成问法规范,
语义向量本身就能覆盖措辞差异,改写无空间。真实学员问法口语化/碎片化
(实测使其 HitRate 低 ≈18pp),这才是改写的潜在主战场。

方法:对 eval_real_style_56.json 的 46 条技术条目,每条跑:
  单次检索(纯向量 top_k=5) vs agent 检索-自省-改写循环(并集合并),
  逐条对照命中 → 转移矩阵 + McNemar + 改写救活率。
  自省/改写用本地 0.5B(与线上快慢分层一致),不调云端。

用法(项目根):
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.rewrite_rescue_real_style
"""
import importlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_rag_main = importlib.import_module("2Milvus_RAG_Qa.core.rag_main")
_rag_system = importlib.import_module("2Milvus_RAG_Qa.core.rag_system")
_agent_graph = importlib.import_module("2Milvus_RAG_Qa.core.agent_graph")
_local_llm = importlib.import_module("2Milvus_RAG_Qa.core.local_llm")
_hit = importlib.import_module("2Milvus_RAG_Qa.RAG评测.hit_rate_eval")
_harness = importlib.import_module("2Milvus_RAG_Qa.RAG评测.eval_harness")

_SET = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_real_style_56.json"
_OUT = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "rewrite_rescue_real_style.json"


def main() -> None:
    items = json.loads(_SET.read_text(encoding="utf-8"))
    tech = [x for x in items if x.get("kind") == "tech_qa"]

    vs = _rag_main.init_knowledge_base()
    rag = _rag_system.RAGSystem(vector_store=vs)
    local = _local_llm.LocalLLM()
    agent = _agent_graph.AdaptiveRAG(rag, local_llm=local)

    a = b = c = d = 0
    rewrote = 0
    rescued_with_rewrite = 0
    per_query = []
    for x in tech:
        q = str(x.get("question", ""))
        expected = str(x.get("expected_chunk", ""))
        single_docs = vs.hybrid_search_with_rerank(q, top_k=5)
        s_hit = _hit._first_hit_rank(expected, single_docs) is not None
        loop = agent.retrieve_loop(q)
        agent_docs = [d["text"] for d in loop["docs"]]
        a_hit = _hit._first_hit_rank(expected, agent_docs) is not None
        rw = loop["rewrite_count"]
        if s_hit and a_hit:
            a += 1
        elif s_hit and not a_hit:
            b += 1
        elif not s_hit and a_hit:
            c += 1
            if rw > 0:
                rescued_with_rewrite += 1
        else:
            d += 1
        if rw > 0:
            rewrote += 1
        per_query.append({"question": q, "single_hit": s_hit, "agent_hit": a_hit,
                          "rewrites": rw, "grade_via": loop.get("grade_via", "")})

    n = len(tech)
    result = {
        "set": "eval_real_style_56.json(技术 46 条,口语化/碎片化问法)",
        "num_queries": n,
        "single_hits": a + b,
        "agent_hits": a + c,
        "single_hit_rate": round((a + b) / n, 4) if n else 0.0,
        "agent_hit_rate": round((a + c) / n, 4) if n else 0.0,
        "transition_matrix": {"both_hit": a, "single_hit_agent_miss": b,
                              "single_miss_agent_hit": c, "both_miss": d},
        "mcnemar": {"b": b, "c": c,
                    "p_value": round(_harness.mcnemar_pvalue(b, c), 4),
                    "test": "mcnemar_exact_paired"},
        "paired_bootstrap_95ci_agent_minus_single": [
            round(x, 4) for x in _harness.paired_bootstrap_delta_ci(
                [int(p["agent_hit"]) for p in per_query],
                [int(p["single_hit"]) for p in per_query])],
        "rewrite_analysis": {
            "rewrote_count": rewrote,
            "avg_rewrites": round(sum(p["rewrites"] for p in per_query) / n, 4) if n else 0.0,
            "rescued_with_rewrite": rescued_with_rewrite,
            "rescue_rate_given_rewrite": round(rescued_with_rewrite / rewrote, 4) if rewrote else None,
        },
        "comparison_with_synthetic": {
            "synthetic_453_rescue_rate": "0/45 = 0.0%(agent_compare.json 终版)",
            "note": "口语化子集若救活率 > 0,则改写价值集中在口语化问法;"
                    "若仍为 0,则改写定位=低成本重试+低置信度降级,不宣称检索增益",
        },
        "per_query": per_query,
    }
    _OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_query"},
                     ensure_ascii=False, indent=2))
    print(f"\n报告已写入 {_OUT}")


if __name__ == "__main__":
    main()
