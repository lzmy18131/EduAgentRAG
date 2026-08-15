"""E5 统计严谨性单测:置信区间与显著性判定(纯函数,不加载重依赖)。"""
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_VENV = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PROJECT_PYTHON = Path(
    os.getenv("EDERAG_PYTHON")
    or (_PROJECT_VENV if _PROJECT_VENV.is_file() else sys.executable)
)


def _run(code: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [str(PROJECT_PYTHON), "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )


def test_eval_stats_pure_functions() -> None:
    """Wilson 区间 / bootstrap MRR CI / z 检验在真实环境计算正确。"""
    code = r"""
import importlib
h = importlib.import_module("2Milvus_RAG_Qa.RAG评测.eval_harness")

# 1. Wilson 区间:命中 334/452 → 中心 ~0.739,区间宽度 ~0.081(与手算一致)
low, high = h.wilson_ci(334, 452)
assert low < 334 / 452 < high, (low, high)
assert 0.06 < high - low < 0.10, (low, high)

# 2. Wilson 边界:0 命中与全命中
assert h.wilson_ci(0, 100)[0] == 0.0
assert h.wilson_ci(100, 100)[1] >= 0.9999  # 浮点误差下约为 0.999998

# 3. z 检验:大差异 → p<0.05;同数据 → p=1
assert h.two_prop_z_pvalue(400, 500, 300, 500) < 0.05
assert h.two_prop_z_pvalue(300, 500, 300, 500) == 1.0

# 4. bootstrap MRR CI:全命中 rank1 → 恒 1.0
assert h.bootstrap_mrr_ci([1.0] * 100) == (1.0, 1.0)
# 半命中 rank1/0 → CI 应覆盖 0.5
low2, high2 = h.bootstrap_mrr_ci([1.0] * 50 + [0.0] * 50)
assert low2 < 0.5 < high2, (low2, high2)

# 5. add_stats 补字段
r = h.add_stats({"num_queries": 452, "hits": 334, "hit_rate": 334 / 452},
                reciprocals=[1.0] * 334 + [0.0] * 118)
assert "hit_rate_ci95" in r and "mrr_ci95" in r
assert len(r["hit_rate_ci95"]) == 2 and len(r["mrr_ci95"]) == 2

print("EVAL_STATS_OK")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "EVAL_STATS_OK" in result.stdout


def test_eval_stats_significance_verdict() -> None:
    """显著性判定:小样本微小波动 → 无显著差异;大样本同样差 → 显著。"""
    code = r"""
import importlib
h = importlib.import_module("2Milvus_RAG_Qa.RAG评测.eval_harness")

# n=30 的 3% 波动(1 条差)→ 不应显著
prev = {"num_queries": 30, "hits": 22, "hit_rate": 22 / 30}
cur = {"num_queries": 30, "hits": 23, "hit_rate": 23 / 30}
sig = h._significance_verdict(prev, cur)
assert sig["verdict"] == "无显著差异", sig

# 同样差 1 条但 n=2000 → 仍不显著(比例差只有 0.05%)
prev2 = {"num_queries": 2000, "hits": 1478, "hit_rate": 1478 / 2000}
cur2 = {"num_queries": 2000, "hits": 1479, "hit_rate": 1479 / 2000}
sig2 = h._significance_verdict(prev2, cur2)
assert sig2["verdict"] == "无显著差异", sig2

# n=2000 差 100 条 → 显著
prev3 = {"num_queries": 2000, "hits": 1478, "hit_rate": 1478 / 2000}
cur3 = {"num_queries": 2000, "hits": 1578, "hit_rate": 1578 / 2000}
sig3 = h._significance_verdict(prev3, cur3)
assert sig3["verdict"] == "显著提升", sig3
assert sig3["p_value"] < 0.05

print("SIG_OK")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "SIG_OK" in result.stdout
