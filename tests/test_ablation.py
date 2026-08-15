"""ablation.py 消融实验脚本的离线单元测试。

在子进程中加载真实环境（脚本 import 链会拉起 torch/milvus 等重依赖），
用 Fake VectorStore / Fake LLM 注入，覆盖 rerank-ablation 与 hyde-ablation
两种模式的核心逻辑（HitRate@K / MRR / 分项 diff），以及 HyDE 改写空返回重试。
"""

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


def run_project_python(*args: str, code: str | None = None) -> subprocess.CompletedProcess:
    command = [str(PROJECT_PYTHON)]
    if code is not None:
        command.extend(["-c", code])
    command.extend(args)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )


def test_ablation_import_and_core_logic() -> None:
    """脚本可导入；两种模式核心逻辑在 mock 下计算结果正确。"""
    code = """
import importlib
import types

ablation = importlib.import_module("2Milvus_RAG_Qa.RAG评测.ablation")

MARK = "目标父块内容"

# ── Fake VectorStore：skip_rerank 决定召回顺序，模拟粗排/精排差异 ──
class FakeVS:
    def hybrid_search_with_rerank(self, query, top_k=3, source_filter=None, skip_rerank=False):
        if skip_rerank:
            return [MARK, "其他A", "其他B"]
        return ["其他C", MARK, "其他D"]

# ── 模式1：rerank-ablation ──
golden = [
    {"question": "q1", "expected_chunk": MARK},
    {"question": "q2", "expected_chunk": "不存在的文本"},
]
res = ablation.run_rerank_ablation(FakeVS(), golden, top_k=3)
assert res["num_queries"] == 2
# 粗排：q1 第1位命中 → hit_rate 0.5, mrr 0.5
assert res["paths"]["rrf_coarse"]["hits"] == 1
assert res["paths"]["rrf_coarse"]["hit_rate"] == 0.5
assert abs(res["paths"]["rrf_coarse"]["mrr"] - 0.5) < 1e-9
# 精排：q1 降到第2位命中 → hit_rate 0.5, mrr 0.25
assert res["paths"]["rrf_rerank"]["hits"] == 1
assert abs(res["paths"]["rrf_rerank"]["mrr"] - 0.25) < 1e-9
assert res["per_item"][0]["diff"] is True      # q1 排名变化
assert res["per_item"][1]["diff"] is False     # q2 两路均未命中

# ── Fake LLM：支持 chat.completions.create，按给定序列返回 ──
class FakeLLM:
    def __init__(self, contents):
        self._contents = list(contents)
        self.n = 0
        self.chat = types.SimpleNamespace(completions=self)
    def create(self, **kwargs):
        self.n += 1
        content = self._contents.pop(0) if self._contents else ""
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=content)
            )]
        )

# HyDE 改写：首次空返回 → 自动重试第 2 次成功
retry_llm = FakeLLM(["   ", "假设答案内容"])
assert ablation._hyde_rewrite(retry_llm, "q") == "假设答案内容"
assert retry_llm.n == 2

# ── 模式2：hyde-ablation ──
MARK2 = "命中文本"
class FakeVS2:
    def hybrid_search_with_rerank(self, query, top_k=3, source_filter=None, skip_rerank=False):
        if "假设答案" in query:
            return [MARK2, "x"]
        return ["无关文本", "y"]

golden2 = [{"question": "原始问题", "expected_chunk": MARK2}]
res2 = ablation.run_hyde_ablation(FakeVS2(), FakeLLM(["假设答案内容"]), golden2, top_k=2)
assert res2["paths"]["direct"]["hits"] == 0
assert res2["paths"]["hyde"]["hits"] == 1
assert res2["paths"]["hyde"]["mrr"] == 1.0
assert res2["per_item"][0]["diff"] is True       # 命中状态变化
assert res2["per_item"][0]["hyde_answer"] == "假设答案内容"
assert res2["per_item"][0]["hyde_fallback"] is False

print("ABLATION_OK")
"""
    result = run_project_python(code=code)
    assert result.returncode == 0, result.stderr
    assert "ABLATION_OK" in result.stdout


def test_ablation_cli_contract() -> None:
    """CLI 契约：支持两种 mode，--top-k 默认 5，报告默认写入 ablation_report.json。"""
    source = (
        PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "ablation.py"
    ).read_text(encoding="utf-8")
    assert 'choices=["rerank-ablation", "hyde-ablation"]' in source
    assert '"--top-k"' in source and "default=5" in source
    assert "ablation_report.json" in source
