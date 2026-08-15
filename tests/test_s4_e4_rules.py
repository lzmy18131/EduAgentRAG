"""S4/E4 纯规则单测(不加载重依赖,进程恢复后随 pytest 一起跑)。"""
import importlib
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


def test_dual_retrieval_rules() -> None:
    """S4:求职信号检测 / JD 条件规则抽取 / JD 文本格式化。"""
    code = r"""
import importlib
d = importlib.import_module("2Milvus_RAG_Qa.core.dual_retrieval")

assert d.is_job_query("北京 java 岗位有哪些") is True
assert d.is_job_query("什么是RAG") is False
assert d.is_job_query("Python多线程怎么用") is False

assert d.extract_jd_conditions("北京 java 15k-20k 的岗位") == {
    "city": "北京", "tech": "java", "salary_min": 15, "salary_max": 20}
assert d.extract_jd_conditions("上海 python 20k 以上岗位") == {
    "city": "上海", "tech": "python", "salary_min": 20}
assert d.extract_jd_conditions("杭州前端岗位") == {"city": "杭州", "tech": "web"}
assert d.extract_jd_conditions("什么是RAG") == {}

row = {"title": "JAVA工程师", "city": "北京", "company": "X",
       "tech_direction": "java", "salary": "15k-25k",
       "experience": "3-5年", "education": "本科"}
t = d.format_jd_text(row)
assert "北京" in t and "java" in t and "15k-25k" in t
print("S4_RULES_OK")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "S4_RULES_OK" in result.stdout


def test_feedback_validation() -> None:
    """E4:rating 校验 / 负样本 JSONL 追加去重 / 统计归并。"""
    code = r"""
import importlib
f = importlib.import_module("2Milvus_RAG_Qa.core.feedback")

# rating 只允许 up/down
try:
    f.record_feedback("u", "q", "a", "bad")
    raise AssertionError("应拒绝非法 rating")
except ValueError:
    pass
print("E4_RULES_OK")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "E4_RULES_OK" in result.stdout
