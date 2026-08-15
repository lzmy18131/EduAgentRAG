# -*- coding: utf-8 -*-
"""统一评测 harness:一条命令跑检索评测,输出报告 + 与上次的指标 diff。

用法(项目根目录):
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.eval_harness hit-rate [--golden ...] [--top-k 5]
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.eval_harness agent-hit-rate [--golden ...]
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.eval_harness compare [--golden ...] [--top-k 5]

核心价值:
  - hit-rate:单次检索;agent-hit-rate:检索-自省-改写循环后的最终检索。
  - compare(2026-08-15 整改):同一黄金集上跑"单次 vs agent"配对对比,
    输出逐 query 命中对照、转移矩阵、McNemar 检验、配对 bootstrap CI
    与改写救活率——替代此前错误的"双比例 z 检验"用法。

E5 统计严谨性(整改后口径):
  - 单组比例:Wilson 95% 置信区间(HitRate);MRR 用 bootstrap(2000 次,seed=42)
    百分位区间,重采样单位 = query。
  - 配对对比(同一批 query 的两个版本):McNemar 检验(精确二项)+
    配对 bootstrap CI(before/after 逐条差值),**禁止**双比例 z 检验。
  - 独立对比(不同 query 集合):双比例 z 检验,p<0.05 才宣称"显著提升/下降",
    否则报"无显著差异"。
  - "p>0.05"只表述为"未观察到统计显著差异",不等价于"证明无差异"。

报告:harness_latest.json + harness_history.json(可 diff)。
"""
import argparse
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_hit_rate_mod = importlib.import_module("2Milvus_RAG_Qa.RAG评测.hit_rate_eval")

_DEFAULT_GOLDEN = str(
    _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden_500.json"
)
_REPORT_DIR = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测"
_LATEST = _REPORT_DIR / "harness_latest.json"
_HISTORY = _REPORT_DIR / "harness_history.json"
_COMPARE_OUT = _REPORT_DIR / "agent_compare.json"

_is_hit = _hit_rate_mod._is_hit
_first_hit_rank = _hit_rate_mod._first_hit_rank

# ─────────────────── E5 统计工具(纯函数,无第三方依赖) ───────────────────

def _norm_cdf(x: float) -> float:
    """标准正态 CDF(erf 实现,避免 scipy 依赖)。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例(命中率)的 Wilson 95% 置信区间。"""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _binom_pmf(n: int, k: int, p: float) -> float:
    """P(X=k), X ~ Bin(n, p)。"""
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def mcnemar_pvalue(b: int, c: int) -> float:
    """配对二值数据的 McNemar 检验(精确二项,双侧)。

    Args:
        b: A 命中、B 未命中的 query 数(regression)
        c: A 未命中、B 命中的 query 数(rescue)

    原假设:两版本命中率相同 → 不一致对 (b, c) 服从 p=0.5 的二项分布,
    双侧 p = 2 * P(Bin(b+c, 0.5) <= min(b, c))。
    适用:同一批 query 先后测两个版本(配对样本)。
    """
    total = b + c
    if total == 0:
        return 1.0  # 无不一致对,两个版本逐 query 完全一致
    return min(1.0, 2 * sum(
        _binom_pmf(total, k, 0.5) for k in range(0, min(b, c) + 1)
    ))


def two_prop_z_pvalue(h1: int, n1: int, h2: int, n2: int) -> float:
    """两组命中率的双比例 z 检验双侧 p 值。

    ⚠️ 仅适用于**独立样本**(两组不同的 query 集合)。
    同一批 query 的 before/after 是配对样本,必须用 mcnemar_pvalue。
    """
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = h1 / n1, h2 / n2
    p_pool = (h1 + h2) / (n1 + n2)
    if p_pool <= 0 or p_pool >= 1:
        return 1.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return 2 * (1 - _norm_cdf(abs(z)))


def bootstrap_mrr_ci(
    reciprocals: list[float], n_boot: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """MRR 的 bootstrap 95% 百分位置信区间(固定种子可复现,重采样单位=query)。"""
    n = len(reciprocals)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(reciprocals, k=n)) / n for _ in range(n_boot)
    )
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def paired_bootstrap_delta_ci(
    a: list[float], b: list[float], n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """配对 bootstrap:同一批 query 上两版本指标差的 95% 百分位 CI。

    Args:
        a, b: 逐 query 的指标值(长度一致,按 query 对齐)
    Returns:
        (下限, 均值差, 上限)
    """
    n = len(a)
    if n == 0 or len(a) != len(b):
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    idx = list(range(n))
    deltas = sorted(
        sum(a[i] - b[i] for i in rng.choices(idx, k=n)) / n
        for _ in range(n_boot)
    )
    mean_delta = sum(a[i] - b[i] for i in idx) / n
    return (deltas[int(0.025 * n_boot)], mean_delta, deltas[int(0.975 * n_boot)])


def add_stats(
    result: dict,
    hits: int | None = None,
    reciprocals: list[float] | None = None,
) -> dict:
    """给评测结果补样本数/置信区间字段(原地修改并返回)。"""
    n = int(result.get("num_queries", 0))
    hits = int(result.get("hits", 0)) if hits is None else int(hits)
    low, high = wilson_ci(hits, n)
    result["hit_rate_ci95"] = [round(low, 4), round(high, 4)]
    if reciprocals is not None:
        mlow, mhigh = bootstrap_mrr_ci(list(reciprocals))
        result["mrr_ci95"] = [round(mlow, 4), round(mhigh, 4)]
    return result


def _significance_verdict(prev: dict, result: dict) -> dict | None:
    """两组命中率是否显著不同:p<0.05 才宣称涨跌。

    同一批 query(per_query 齐备)→ McNemar 配对检验;
    否则(不同集合)→ 双比例 z 检验并显式标注 test 类型。
    """
    if not (prev.get("num_queries") and result.get("num_queries")):
        return None
    d = result.get("hit_rate", 0.0) - prev.get("hit_rate", 0.0)
    prev_pq = prev.get("per_query") or []
    cur_pq = result.get("per_query") or []
    if prev_pq and cur_pq and len(prev_pq) == len(cur_pq):
        # 逐 query 对齐(以问题文本为 key,避免顺序差异)
        prev_hit = {str(x.get("question", "")): bool(x.get("hit")) for x in prev_pq}
        cur_hit = {str(x.get("question", "")): bool(x.get("hit")) for x in cur_pq}
        common = [q for q in prev_hit if q in cur_hit]
        if len(common) >= 2:
            b = sum(1 for q in common if prev_hit[q] and not cur_hit[q])
            c = sum(1 for q in common if not prev_hit[q] and cur_hit[q])
            p = mcnemar_pvalue(b, c)
            verdict = "无显著差异" if p >= 0.05 else ("显著提升" if d > 0 else "显著下降")
            return {
                "p_value": round(p, 4), "verdict": verdict,
                "delta_hit_rate": round(d, 4), "test": "mcnemar_paired",
                "discordant_pairs": {"prev_hit_cur_miss": b, "prev_miss_cur_hit": c},
            }
    p = two_prop_z_pvalue(
        int(prev["hits"]), int(prev["num_queries"]),
        int(result["hits"]), int(result["num_queries"]),
    )
    verdict = "无显著差异" if p >= 0.05 else ("显著提升" if d > 0 else "显著下降")
    return {
        "p_value": round(p, 4), "verdict": verdict,
        "delta_hit_rate": round(d, 4), "test": "two_prop_z_unpaired",
    }


# ─────────────────── 评测入口 ───────────────────

def _load_golden(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_hit_rate(golden: list[dict], top_k: int) -> dict:
    """经典单次检索评测(复用 hit_rate_eval)。"""
    result = _hit_rate_mod.run_eval(golden, top_k)
    reciprocals = result.pop("reciprocals", [])
    return add_stats(result, reciprocals=reciprocals)


def run_agent_hit_rate(rag, agent, golden: list[dict]) -> dict:
    """agent 检索-自省-改写循环后的最终检索评测(逐 query 结果落盘)。"""
    hits = 0
    reciprocals: list[float] = []
    rewrites = 0
    per_query: list[dict] = []
    for item in golden:
        question = str(item.get("question", ""))
        expected = str(item.get("expected_chunk", ""))
        loop = agent.retrieve_loop(question)
        docs = [d["text"] for d in loop["docs"]]
        rewrites += loop["rewrite_count"]
        rank = _first_hit_rank(expected, docs)
        per_query.append({
            "question": question,
            "hit": rank is not None,
            "rank": rank,
            "rewrites": loop["rewrite_count"],
            "grade_via": loop.get("grade_via", ""),
        })
        if rank is not None:
            hits += 1
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)
    total = len(golden)
    result = {
        "num_queries": total,
        "hits": hits,
        "hit_rate": hits / total if total else 0.0,
        "mrr": sum(reciprocals) / total if total else 0.0,
        "avg_rewrites": rewrites / total if total else 0.0,
        "per_query": per_query,
    }
    return add_stats(result, reciprocals=reciprocals)


def run_compare(rag, agent, golden: list[dict], top_k: int) -> dict:
    """配对对比:同一黄金集上"单次检索 vs agent 循环"(整改新增)。

    输出:
      - 逐 query 对照(单次 hit / agent hit / 改写次数)
      - 转移矩阵 a/b/c/d:
          a = 两者都命中;b = 单次命中 agent 未命中(regression)
          c = 单次未命中 agent 命中(rescue);d = 两者都未命中
      - McNemar 检验(配对,精确二项)
      - 命中率差与 MRR 差的配对 bootstrap 95% CI
      - 改写救活率(发生过改写的 query 中,未命中→命中的比例)
    """
    vs = rag._vs
    single_hits: list[int] = []
    agent_hits: list[int] = []
    single_rr: list[float] = []
    agent_rr: list[float] = []
    per_query: list[dict] = []
    a = b = c = d = 0
    rescued_with_rewrite = 0
    rewrote_count = 0

    for item in golden:
        question = str(item.get("question", ""))
        expected = str(item.get("expected_chunk", ""))
        # 单次:与 harness hit-rate 同口径(纯向量检索,不触发 JD 双路)
        single_docs = vs.hybrid_search_with_rerank(question, top_k=top_k)
        s_rank = _first_hit_rank(expected, single_docs)
        # agent:检索-自省-改写循环(含求职意图双路召回)
        loop = agent.retrieve_loop(question)
        agent_docs = [d["text"] for d in loop["docs"]]
        a_rank = _first_hit_rank(expected, agent_docs)
        rewrites = loop["rewrite_count"]

        s_hit = s_rank is not None
        a_hit = a_rank is not None
        single_hits.append(int(s_hit))
        agent_hits.append(int(a_hit))
        single_rr.append(1.0 / s_rank if s_hit else 0.0)
        agent_rr.append(1.0 / a_rank if a_hit else 0.0)
        if s_hit and a_hit:
            a += 1
        elif s_hit and not a_hit:
            b += 1
        elif not s_hit and a_hit:
            c += 1
            if rewrites > 0:
                rescued_with_rewrite += 1
        else:
            d += 1
        if rewrites > 0:
            rewrote_count += 1
        per_query.append({
            "question": question,
            "single_hit": s_hit, "single_rank": s_rank,
            "agent_hit": a_hit, "agent_rank": a_rank,
            "rewrites": rewrites,
        })

    n = len(golden)
    hr_low, hr_mean, hr_high = paired_bootstrap_delta_ci(agent_hits, single_hits)
    mr_low, mr_mean, mr_high = paired_bootstrap_delta_ci(agent_rr, single_rr)
    p = mcnemar_pvalue(b, c)
    single_hr = sum(single_hits) / n if n else 0.0
    agent_hr = sum(agent_hits) / n if n else 0.0
    result = {
        "mode": "compare_single_vs_agent",
        "top_k": top_k,
        "num_queries": n,
        "single": {
            "hits": sum(single_hits), "hit_rate": round(single_hr, 4),
            "mrr": round(sum(single_rr) / n, 4) if n else 0.0,
        },
        "agent": {
            "hits": sum(agent_hits), "hit_rate": round(agent_hr, 4),
            "mrr": round(sum(agent_rr) / n, 4) if n else 0.0,
        },
        "transition_matrix": {
            "both_hit": a, "single_hit_agent_miss": b,
            "single_miss_agent_hit": c, "both_miss": d,
        },
        "mcnemar": {
            "b": b, "c": c, "p_value": round(p, 4),
            "note": "McNemar 精确二项检验(配对样本);p<0.05 才宣称显著差异",
        },
        "paired_bootstrap_95ci": {
            "agent_minus_single_hit_rate": [round(hr_low, 4), round(hr_mean, 4), round(hr_high, 4)],
            "agent_minus_single_mrr": [round(mr_low, 4), round(mr_mean, 4), round(mr_high, 4)],
            "n_boot": 2000, "seed": 42,
        },
        "rewrite_analysis": {
            "rewrote_count": rewrote_count,
            "avg_rewrites": round(sum(x["rewrites"] for x in per_query) / n, 4) if n else 0.0,
            "rescue_total": c,
            "rescued_with_rewrite": rescued_with_rewrite,
            "rescue_rate_given_rewrite": round(rescued_with_rewrite / rewrote_count, 4) if rewrote_count else None,
            "note": "改写救活率=发生改写的 query 中由未命中变为命中的比例;净变化还需看 regression(b)",
        },
        "per_query": per_query,
    }
    _COMPARE_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def save_and_diff(result: dict) -> dict:
    """写 latest + 追加 history,返回与上次的 diff(含显著性判定)。"""
    prev = None
    if _HISTORY.exists():
        history = json.loads(_HISTORY.read_text(encoding="utf-8"))
        if history:
            prev = history[-1]
    result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    history = json.loads(_HISTORY.read_text(encoding="utf-8")) if _HISTORY.exists() else []
    history.append(result)
    _HISTORY.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    _LATEST.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    diff = {}
    if prev:
        for k in ("hit_rate", "mrr", "avg_rewrites"):
            if k in prev and k in result:
                diff[k] = round(result[k] - prev[k], 4)
        sig = _significance_verdict(prev, result)
        if sig:
            diff["hit_rate_significance"] = sig
    return diff


def _init_rag_agent():
    rag_main = importlib.import_module("2Milvus_RAG_Qa.core.rag_main")
    rag_system = importlib.import_module("2Milvus_RAG_Qa.core.rag_system")
    agent_graph = importlib.import_module("2Milvus_RAG_Qa.core.agent_graph")
    vs = rag_main.init_knowledge_base()
    rag = rag_system.RAGSystem(vector_store=vs)
    agent = agent_graph.AdaptiveRAG(rag)
    return rag, agent


def main() -> None:
    parser = argparse.ArgumentParser(description="统一评测 harness")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_hr = sub.add_parser("hit-rate", help="单次检索命中率")
    p_hr.add_argument("--golden", default=_DEFAULT_GOLDEN)
    p_hr.add_argument("--top-k", type=int, default=5)
    p_ar = sub.add_parser("agent-hit-rate", help="agent 循环检索命中率")
    p_ar.add_argument("--golden", default=_DEFAULT_GOLDEN)
    p_cp = sub.add_parser("compare", help="单次 vs agent 配对对比(McNemar+转移矩阵)")
    p_cp.add_argument("--golden", default=_DEFAULT_GOLDEN)
    p_cp.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    golden = _load_golden(args.golden)
    if args.cmd == "hit-rate":
        result = run_hit_rate(golden, args.top_k)
        result["mode"] = "hit_rate"
        result["top_k"] = args.top_k
        diff = save_and_diff(result)
        print(json.dumps({"result": result, "diff_vs_prev": diff}, ensure_ascii=False, indent=2))
    elif args.cmd == "agent-hit-rate":
        rag, agent = _init_rag_agent()
        result = run_agent_hit_rate(rag, agent, golden)
        result["mode"] = "agent_hit_rate"
        diff = save_and_diff(result)
        print(json.dumps({"result": result, "diff_vs_prev": diff}, ensure_ascii=False, indent=2))
    else:  # compare
        rag, agent = _init_rag_agent()
        result = run_compare(rag, agent, golden, args.top_k)
        print(json.dumps({k: v for k, v in result.items() if k != "per_query"},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
