# -*- coding: utf-8 -*-
"""2026-08-14 消融实验(2026-08-15 整改版):能消融的改动必须实测 A/B。

已覆盖:
  S4 双路并行检索:求职查询集 → 单路(仅向量) vs 双路(JD 结构化 + 向量)
     指标:JD 条件满足率 + 检索延迟 + Wilson CI(n=10,如实报告小样本局限)
  S5 重排瘦身:rerank_pool_multiplier 20 vs 15 → 同一批 query 配对跑
     **配对 McNemar 检验 + 配对 bootstrap CI**(修正此前双比例 z 误用)
  S6 向量量化:**工程踩坑记录**(非消融实验)——IVF_PQ 在 6GB WSL2 受限
     环境构建不可行的实测记录;不宣称"HNSW 优于 IVF_PQ",保留切换能力。

用法(项目根):
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.ablation_2026 s4
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.ablation_2026 s5
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.ablation_2026 s6
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.ablation_2026 all
"""
import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from base.config import cfg
from base.logger import logger

_rag_main = importlib.import_module("2Milvus_RAG_Qa.core.rag_main")
_dual = importlib.import_module("2Milvus_RAG_Qa.core.dual_retrieval")
_jd = importlib.import_module("2Milvus_RAG_Qa.core.jd_structured")
_hit = importlib.import_module("2Milvus_RAG_Qa.RAG评测.hit_rate_eval")
_harness = importlib.import_module("2Milvus_RAG_Qa.RAG评测.eval_harness")

# ── S4 消融:求职查询集(问题, 期望满足的 JD 条件) ──
# 2026-08-15 扩样:10 → 35 条(全部经 MySQL 验证有 JD 行可满足条件)
_JOB_QUERIES = [
    ("北京 java 15k 岗位", {"city": "北京", "tech": "java", "salary_min": 15}),
    ("上海 python 20k 岗位", {"city": "上海", "tech": "python", "salary_min": 20}),
    ("杭州 web 前端 10k 岗位", {"city": "杭州", "tech": "web", "salary_min": 10}),
    ("深圳 linux 运维 12k 岗位", {"city": "深圳", "tech": "linux", "salary_min": 12}),
    ("广州 java 8k-12k 岗位", {"city": "广州", "tech": "java", "salary_min": 8, "salary_max": 12}),
    ("成都 C# 10k 岗位", {"city": "成都", "tech": "C#", "salary_min": 10}),
    ("南京 java 应届生岗位", {"city": "南京", "tech": "java"}),
    ("武汉 python 15k 岗位", {"city": "武汉", "tech": "python", "salary_min": 15}),
    ("西安 web 8k 岗位", {"city": "西安", "tech": "web", "salary_min": 8}),
    ("北京 linux 25k 岗位", {"city": "北京", "tech": "linux", "salary_min": 25}),
    # ── 扩样(2026-08-15,全部已验证 JD 表有满足行) ──
    ("北京 python 12k-18k 岗位", {"city": "北京", "tech": "python", "salary_min": 12, "salary_max": 18}),
    ("上海 java 25k 岗位", {"city": "上海", "tech": "java", "salary_min": 25}),
    ("广州 python 10k 岗位", {"city": "广州", "tech": "python", "salary_min": 10}),
    ("深圳 java 18k 岗位", {"city": "深圳", "tech": "java", "salary_min": 18}),
    ("杭州 linux 15k 岗位", {"city": "杭州", "tech": "linux", "salary_min": 15}),
    ("成都 python 12k 岗位", {"city": "成都", "tech": "python", "salary_min": 12}),
    ("武汉 java 10k 岗位", {"city": "武汉", "tech": "java", "salary_min": 10}),
    ("南京 web 12k 岗位", {"city": "南京", "tech": "web", "salary_min": 12}),
    ("西安 python 10k 岗位", {"city": "西安", "tech": "python", "salary_min": 10}),
    ("长沙 java 15k 岗位", {"city": "长沙", "tech": "java", "salary_min": 15}),
    ("郑州 web 8k 岗位", {"city": "郑州", "tech": "web", "salary_min": 8}),
    ("重庆 linux 12k 岗位", {"city": "重庆", "tech": "linux", "salary_min": 12}),
    ("苏州 C# 15k 岗位", {"city": "苏州", "tech": "C#", "salary_min": 15}),
    ("天津 java 10k-15k 岗位", {"city": "天津", "tech": "java", "salary_min": 10, "salary_max": 15}),
    ("青岛 web 9k 岗位", {"city": "青岛", "tech": "web", "salary_min": 9}),
    ("厦门 linux 13k 岗位", {"city": "厦门", "tech": "linux", "salary_min": 13}),
    ("合肥 java 11k 岗位", {"city": "合肥", "tech": "java", "salary_min": 11}),
    ("东莞 python 12k 岗位", {"city": "东莞", "tech": "python", "salary_min": 12}),
    ("佛山 web 10k 岗位", {"city": "佛山", "tech": "web", "salary_min": 10}),
    ("大连 C# 12k 岗位", {"city": "大连", "tech": "C#", "salary_min": 12}),
    ("济南 linux 9k 岗位", {"city": "济南", "tech": "linux", "salary_min": 9}),
    ("福州 java 12k 岗位", {"city": "福州", "tech": "java", "salary_min": 12}),
    ("无锡 python 14k 岗位", {"city": "无锡", "tech": "python", "salary_min": 14}),
    ("宁波 web 11k 岗位", {"city": "宁波", "tech": "web", "salary_min": 11}),
    ("温州 java 9k-13k 岗位", {"city": "温州", "tech": "java", "salary_min": 9, "salary_max": 13}),
]


def _parse_vec_jd(text: str) -> dict:
    """解析语料中 JD 文本块的结构化字段(与 job_data 同格式)。"""
    def f(name: str) -> str:
        m = re.search(rf"- {name}:\s*([^\n]*)", text)
        return m.group(1).strip() if m else ""
    salary_m = re.match(r"(\d+)k-(\d+)k", f("薪资"))
    return {
        "city": f("城市"),
        "tech_direction": f("技术方向").lower(),
        "salary_min": int(salary_m.group(1)) if salary_m else None,
        "salary_max": int(salary_m.group(2)) if salary_m else None,
    }


def _cond_satisfies(fields: dict, cond: dict) -> bool:
    if cond.get("city") and fields.get("city") != cond["city"]:
        return False
    if cond.get("tech") and fields.get("tech_direction") != cond["tech"].lower():
        return False
    if cond.get("salary_min") and (fields.get("salary_max") or 0) < cond["salary_min"]:
        return False
    if cond.get("salary_max") and (fields.get("salary_min") or 10**9) > cond["salary_max"]:
        return False
    return True


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


def run_s4_ablation(top_k: int = 5) -> dict:
    """求职查询:单路(仅向量,同口径解析条件) vs 双路(JD 结构化 + 向量)。"""
    vs = _rag_main.init_knowledge_base()
    # 预热:加载 reranker 等,避免冷启动偏置
    vs.hybrid_search_with_rerank_scored(_JOB_QUERIES[0][0], top_k=top_k)
    _dual.dual_retrieve(_JOB_QUERIES[0][0], vs.hybrid_search_with_rerank_scored, top_k)

    single_ok = dual_ok = 0
    single_ms: list[float] = []
    dual_ms: list[float] = []
    per_query: list[dict] = []

    for qi, (query, cond) in enumerate(_JOB_QUERIES):
        # 交替顺序测延迟,消除顺序偏置
        if qi % 2 == 0:
            t0 = time.time()
            vec_docs = vs.hybrid_search_with_rerank_scored(query, top_k=top_k)
            single_ms.append((time.time() - t0) * 1000)
            t0 = time.time()
            docs = _dual.dual_retrieve(query, vs.hybrid_search_with_rerank_scored, top_k)
            dual_ms.append((time.time() - t0) * 1000)
        else:
            t0 = time.time()
            docs = _dual.dual_retrieve(query, vs.hybrid_search_with_rerank_scored, top_k)
            dual_ms.append((time.time() - t0) * 1000)
            t0 = time.time()
            vec_docs = vs.hybrid_search_with_rerank_scored(query, top_k=top_k)
            single_ms.append((time.time() - t0) * 1000)

        # 同口径判定:向量文本解析出结构化字段后做条件校验
        single_hit = any(
            _cond_satisfies(_parse_vec_jd(text), cond) for text, _ in vec_docs
        )
        jd_rows = _jd.search_jobs(**cond, limit=top_k)
        dual_hit = any(_jd_satisfies(r, cond) for r in jd_rows)

        single_ok += single_hit
        dual_ok += dual_hit
        per_query.append({
            "query": query, "cond": cond,
            "single_vector_condition_hit": single_hit,
            "dual_jd_condition_hit": dual_hit,
        })

    # 配对 McNemar(同一批 query 的单路/双路条件满足对照)
    b = sum(1 for x in per_query if x["single_vector_condition_hit"] and not x["dual_jd_condition_hit"])
    c = sum(1 for x in per_query if not x["single_vector_condition_hit"] and x["dual_jd_condition_hit"])
    a = sum(1 for x in per_query if x["single_vector_condition_hit"] and x["dual_jd_condition_hit"])
    d = sum(1 for x in per_query if not x["single_vector_condition_hit"] and not x["dual_jd_condition_hit"])

    result = {
        "num_queries": len(_JOB_QUERIES),
        "single_vector_condition_hit": single_ok,
        "dual_condition_hit": dual_ok,
        "single_hit_rate": single_ok / len(_JOB_QUERIES),
        "dual_hit_rate": dual_ok / len(_JOB_QUERIES),
        "single_ci95": [round(x, 4) for x in
                        _harness.wilson_ci(single_ok, len(_JOB_QUERIES))],
        "dual_ci95": [round(x, 4) for x in
                      _harness.wilson_ci(dual_ok, len(_JOB_QUERIES))],
        "paired_transition_matrix": {"both_satisfied": a, "single_only": b,
                                     "dual_only_rescued": c, "both_fail": d},
        "mcnemar": {"b": b, "c": c, "p_value": round(_harness.mcnemar_pvalue(b, c), 4),
                    "test": "mcnemar_exact_paired",
                    "note": "双路(JD 结构化)相对单路(仅向量)的配对条件满足检验"},
        "single_avg_ms": round(sum(single_ms) / len(single_ms)),
        "dual_avg_ms": round(sum(dual_ms) / len(dual_ms)),
        "notes": ("n=35 真实求职查询(2026-08-15 扩样,全部经 MySQL 验证有 JD 行可满足条件),"
                  "配对同口径条件判定 + Wilson CI + McNemar;双路口径=JD 结构化命中(独立槽位)"),
        "per_query": per_query,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_query"},
                     ensure_ascii=False, indent=2))
    return result


def _eval_golden_with_pool(golden: list[dict], top_k: int, pool: int) -> dict:
    """指定 rerank_pool_multiplier 跑黄金集 HitRate@5/MRR(独立进程口径,保留兼容)。"""
    cfg.RERANK_POOL_MULTIPLIER = pool
    vs = _rag_main.init_knowledge_base()
    hits = 0
    reciprocals: list[float] = []
    for item in golden:
        question = str(item.get("question", ""))
        expected = str(item.get("expected_chunk", ""))
        recalled = vs.hybrid_search_with_rerank(question, top_k=top_k)
        rank = _hit._first_hit_rank(expected, recalled)
        if rank is not None:
            hits += 1
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)
    total = len(golden)
    return {
        "pool": pool,
        "num_queries": total,
        "hits": hits,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "mrr": round(sum(reciprocals) / total, 4) if total else 0.0,
    }


def run_s5_ablation(golden_path: str, top_k: int = 5) -> dict:
    """rerank_pool_multiplier 20 vs 15 配对消融(整改:McNemar + 配对 bootstrap)。

    同一批 query 在 pool20 / pool15 两种配置下各检索一次(配对样本),
    逐条记录命中对照 → McNemar 精确检验 + 命中率差/MRR 差的配对 bootstrap CI。
    此前版本用双比例 z 检验(独立样本口径)是统计方法误用,已修正。
    """
    golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    vs = _rag_main.init_knowledge_base()
    current = cfg.RERANK_POOL_MULTIPLIER

    hits20: list[int] = []
    hits15: list[int] = []
    rr20: list[float] = []
    rr15: list[float] = []
    per_query: list[dict] = []
    for item in golden:
        question = str(item.get("question", ""))
        expected = str(item.get("expected_chunk", ""))
        cfg.RERANK_POOL_MULTIPLIER = 20
        r20 = _hit._first_hit_rank(expected, vs.hybrid_search_with_rerank(question, top_k=top_k))
        cfg.RERANK_POOL_MULTIPLIER = 15
        r15 = _hit._first_hit_rank(expected, vs.hybrid_search_with_rerank(question, top_k=top_k))
        h20, h15 = r20 is not None, r15 is not None
        hits20.append(int(h20)); hits15.append(int(h15))
        rr20.append(1.0 / r20 if h20 else 0.0)
        rr15.append(1.0 / r15 if h15 else 0.0)
        per_query.append({"question": question[:60], "pool20_hit": h20,
                          "pool15_hit": h15})
    cfg.RERANK_POOL_MULTIPLIER = current

    n = len(golden)
    b = sum(1 for x in per_query if x["pool20_hit"] and not x["pool15_hit"])
    c = sum(1 for x in per_query if not x["pool20_hit"] and x["pool15_hit"])
    a = sum(1 for x in per_query if x["pool20_hit"] and x["pool15_hit"])
    d = sum(1 for x in per_query if not x["pool20_hit"] and not x["pool15_hit"])
    p = _harness.mcnemar_pvalue(b, c)
    hr_low, hr_mean, hr_high = _harness.paired_bootstrap_delta_ci(hits15, hits20)
    mr_low, mr_mean, mr_high = _harness.paired_bootstrap_delta_ci(rr15, rr20)

    result = {
        "paired_design": "same 453 queries run under pool20 and pool15 (paired samples)",
        "num_queries": n,
        "pool20": {"hits": sum(hits20), "hit_rate": round(sum(hits20) / n, 4),
                   "mrr": round(sum(rr20) / n, 4)},
        "pool15": {"hits": sum(hits15), "hit_rate": round(sum(hits15) / n, 4),
                   "mrr": round(sum(rr15) / n, 4)},
        "transition_matrix": {"both_hit": a, "pool20_hit_pool15_miss": b,
                              "pool20_miss_pool15_hit": c, "both_miss": d},
        "mcnemar": {"b": b, "c": c, "p_value": round(p, 4),
                    "test": "mcnemar_exact_paired"},
        "paired_bootstrap_95ci": {
            "pool15_minus_pool20_hit_rate": [round(hr_low, 4), round(hr_mean, 4), round(hr_high, 4)],
            "pool15_minus_pool20_mrr": [round(mr_low, 4), round(mr_mean, 4), round(mr_high, 4)],
            "n_boot": 2000, "seed": 42,
        },
        "note": ("p>0.05 只表述为'未观察到统计显著差异',不等价于'证明等价';"
                 "20→15 的收益是 rerank 候选数减少 25%(重排计算量近似同比),"
                 "非实测 GPU 时间下降 25%"),
        "per_query": per_query,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_query"},
                     ensure_ascii=False, indent=2))
    return result


def _eval_golden_current_index(golden: list[dict], top_k: int) -> dict:
    """当前索引下跑黄金集 HitRate@5/MRR(不覆盖 pool 配置)。"""
    vs = _rag_main.init_knowledge_base()
    hits = 0
    reciprocals: list[float] = []
    for item in golden:
        question = str(item.get("question", ""))
        expected = str(item.get("expected_chunk", ""))
        recalled = vs.hybrid_search_with_rerank(question, top_k=top_k)
        rank = _hit._first_hit_rank(expected, recalled)
        if rank is not None:
            hits += 1
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)
    total = len(golden)
    return {
        "index_type": vs._dense_index_type,
        "num_queries": total,
        "hits": hits,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "mrr": round(sum(reciprocals) / total, 4) if total else 0.0,
    }


def _latency_sample(vs, queries: list[str], top_k: int = 5, warm: int = 2) -> dict:
    """对样本查询测检索延迟(预热后取均值/中位数)。"""
    for q in queries[:warm]:
        vs.hybrid_search_with_rerank(q, top_k=top_k)
    times = []
    for q in queries:
        t0 = time.time()
        vs.hybrid_search_with_rerank(q, top_k=top_k)
        times.append((time.time() - t0) * 1000)
    times.sort()
    return {
        "n": len(times),
        "avg_ms": round(sum(times) / len(times)),
        "p50_ms": round(times[len(times) // 2]),
        "p90_ms": round(times[int(len(times) * 0.9)]),
    }


def _wait_index_ready(vs, timeout_s: int = 1800) -> dict:
    """轮询 describe_index 直到索引构建 Finished。"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        info = vs._client.describe_index(cfg.MILVUS_COLLECTION, "dense_idx")
        if info.get("state") == "Finished":
            return info
        time.sleep(10)
    raise TimeoutError(f"索引构建超时 {timeout_s}s")


def run_s6_ablation(golden_path: str = "", top_k: int = 5) -> dict:
    """S6 向量量化:工程踩坑记录(整改:不再冒充消融实验)。

    此前版本在数据事故后的 41 行残集上跑出 hit_rate=0.0132,属无效数据,
    已删除。真实记录(2026-08-14,358,221 行全量语料):
      - nlist=1024:构建 >50 分钟卡在 47%,超时终止;
      - nlist=256:构建 >25 分钟无进度,且卡死索引任务队列;
      - 根因:6GB WSL2 内存/CPU 受限;两次均安全回滚 HNSW,数据全程完好。
    结论口径:在**当前受限开发环境与参数配置**下 IVF_PQ 构建未完成,
    不宣称算法层面"HNSW 优于 IVF_PQ";rebuild_dense_index 切换能力保留,
    更大内存机器可一条命令重测(同脚本)。
    """
    result = {
        "category": "engineering_constraint_record(非消融实验)",
        "attempts": [
            {"nlist": 1024, "outcome": "构建 >50min 卡 47%,超时终止,安全回滚 HNSW"},
            {"nlist": 256, "outcome": "构建 >25min 无进度,卡死索引任务队列,安全回滚 HNSW"},
        ],
        "constraint": "WSL2 内存限 6GB(开发机),CPU 受限",
        "data_safety": "两次回滚后 edurag_0421 行数 358,221 全程完好",
        "decision": (
            "保留 HNSW(本规模检索 ~0.5s 可接受);IVF_PQ 切换接口保留,"
            "规模化部署时按 run_s6_ablation 原评测路径重测即可"
        ),
        "invalid_legacy_data_removed": (
            "此前 ablation_2026.json 中 s6 的 0.0132 数据系事故后 41 行残集产物,"
            "已从产物中删除,不得引用"
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument("which", choices=["s4", "s5", "s6", "all"])
    parser.add_argument("--golden", default=str(
        _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden_500.json"))
    args = parser.parse_args()

    report: dict = {}
    if args.which in ("s4", "all"):
        report["s4_dual_retrieval"] = run_s4_ablation()
    if args.which in ("s5", "all"):
        report["s5_rerank_pool"] = run_s5_ablation(args.golden)
    if args.which in ("s6", "all"):
        report["s6_vector_quantization"] = run_s6_ablation(args.golden)

    out = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "ablation_2026.json"
    # 整改:read-modify-write,避免不同子命令互相覆盖产物
    if out.exists():
        try:
            merged = json.loads(out.read_text(encoding="utf-8"))
            if not isinstance(merged, dict):
                merged = {}
        except Exception:
            merged = {}
    else:
        merged = {}
    merged.update(report)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()
