# -*- coding: utf-8 -*-
"""Grade 模型质量评测(整改版):0.5B 自省判决的混淆矩阵 + 混沌区仲裁精度。

回答:"快模型做自省,误判率多少?"
方法:全量黄金 query → agent 检索(与线上同口径:向量路 top1 重排分,
JD 槽位不参与门控)→ 以"期望 chunk 是否命中"为真值(hit=文档充分)
→ 三级门控判决 → 混淆矩阵:
  TP: 命中且判"充分"   FN: 命中却判"不充分"(→ 触发无谓改写,浪费一轮)
  FP: 未命中却判"充分"(→ 漏掉改写机会)
  TN: 未命中且判"不充分"
混沌区(0.4, 0.7)由 0.5B 仲裁,单独报告其准确率 + Wilson 95% CI——
不再只报 7 条小样本的点估计。

用法(项目根):
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.grade_quality [--num N]
  默认 N=全量 453 条(混沌区样本随全量评测显著扩大)。
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from base.logger import logger

_rag_main = importlib.import_module("2Milvus_RAG_Qa.core.rag_main")
_rag_system = importlib.import_module("2Milvus_RAG_Qa.core.rag_system")
_agent_graph = importlib.import_module("2Milvus_RAG_Qa.core.agent_graph")
_local_llm = importlib.import_module("2Milvus_RAG_Qa.core.local_llm")
_hit = importlib.import_module("2Milvus_RAG_Qa.RAG评测.hit_rate_eval")
_harness = importlib.import_module("2Milvus_RAG_Qa.RAG评测.eval_harness")

_DEFAULT_GOLDEN = str(
    _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden_500.json"
)
_OUT = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "grade_quality.json"


def _rate_wilson(hits: int, n: int) -> dict:
    low, high = _harness.wilson_ci(hits, n)
    return {"hits": hits, "n": n, "rate": round(hits / n, 4) if n else 0.0,
            "ci95": [round(low, 4), round(high, 4)]}


def main() -> None:
    parser = argparse.ArgumentParser(description="0.5B 自省模型混淆矩阵评测(全量)")
    parser.add_argument("--num", type=int, default=None,
                        help="评测条数;默认 None=全量黄金集")
    parser.add_argument("--golden", default=_DEFAULT_GOLDEN)
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    if args.num:
        golden = golden[: args.num]

    vs = _rag_main.init_knowledge_base()
    rag = _rag_system.RAGSystem(vector_store=vs)
    local = _local_llm.LocalLLM()  # 0.5B 快模型
    agent = _agent_graph.AdaptiveRAG(rag, local_llm=local)

    tp = fn = fp = tn = 0
    gate_pass = gate_pass_wrong = 0
    gate_fail = gate_fail_wrong = 0
    chaos_n = chaos_correct = 0
    rows = []
    for item in golden:
        q = str(item.get("question", ""))
        expected = str(item.get("expected_chunk", ""))
        docs = agent._retrieve({"query": q, "original_query": q, "docs": [],
                                "grade": "", "rewrite_count": 0, "answer": ""})["docs"]
        vec_docs = [d for d in docs if d.get("source") != "jd"]
        texts = [d["text"] for d in vec_docs]
        is_hit = _hit._first_hit_rank(expected, texts) is not None  # 真值

        if not vec_docs:
            # 仅 JD 命中或无文档:与 agent._grade 同口径
            verdict = "充分" if any(d.get("source") == "jd" for d in docs) else "不充分"
            via = "jd_struct_only" if any(d.get("source") == "jd" for d in docs) else "no_docs"
        else:
            top1 = max(float(d.get("score", 0.0)) for d in vec_docs)
            if top1 >= _agent_graph.GRADE_SKIP_THRESHOLD:
                verdict = "充分"  # 一级门控直过
                via = "score_gate_pass"
                gate_pass += 1
                if not is_hit:
                    gate_pass_wrong += 1  # 门控放行但实际未命中
            elif top1 <= _agent_graph.GRADE_AUTO_FAIL:
                verdict = "不充分"  # 二级门控直拒(不调 LLM)
                via = "score_gate_fail"
                gate_fail += 1
                if is_hit:
                    gate_fail_wrong += 1  # 门控直拒但实际命中(无谓改写)
            else:
                # 混沌区:LLM 仲裁(与 agent._grade 同款 prompt)
                chaos_n += 1
                docs_text = "\n\n".join(f"[{i + 1}] {d['text'][:300]}" for i, d in enumerate(docs))
                prompt = (
                    "你是检索质量评估器。判断以下检索到的文档是否足以回答用户问题。\n"
                    '只输出 JSON:{"verdict": "充分"} 或 {"verdict": "不充分"}。\n\n'
                    f"用户问题:{q}\n\n检索到的文档:\n{docs_text}\n"
                )
                verdict = agent._parse_verdict(local.generate(prompt, max_tokens=32))
                via = "llm_chaos"
                if (verdict == "充分") == is_hit:
                    chaos_correct += 1

        sufficient = verdict == "充分"
        if is_hit and sufficient:
            tp += 1
        elif is_hit and not sufficient:
            fn += 1
        elif not is_hit and sufficient:
            fp += 1
        else:
            tn += 1
        rows.append({"q": q[:40], "hit": is_hit, "verdict": verdict, "via": via})

    n = len(rows)
    correct = tp + tn
    result = {
        "num_queries": n,
        "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "accuracy": round(correct / n, 4) if n else 0,
        "accuracy_ci95": [round(x, 4) for x in _harness.wilson_ci(correct, n)],
        "fn_rate": round(fn / n, 4) if n else 0,   # 命中却判不充分 → 无谓改写
        "fp_rate": round(fp / n, 4) if n else 0,   # 未命中却判充分 → 漏改写
        "gate_pass": _rate_wilson(gate_pass, n),
        "gate_pass_and_miss": _rate_wilson(gate_pass_wrong, gate_pass),
        "gate_fail": _rate_wilson(gate_fail, n),
        "gate_fail_but_hit": _rate_wilson(gate_fail_wrong, gate_fail),
        "chaos_zone": _rate_wilson(chaos_n, n),
        "chaos_llm_accuracy": _rate_wilson(chaos_correct, chaos_n),
        "notes": (
            "真值=期望 chunk 是否命中向量路召回;门控分数=向量路重排分 top1,"
            "JD 槽位不参与;混沌区仲裁由本地 Qwen2.5-0.5B 执行(云端回退)。"
        ),
        "rows": rows,
    }
    _OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
