"""整改(2026-08-15)配套单测:统计方法/意图路由/门控隔离/上下文规则/缓存隔离。

纯规则与纯函数,不加载重依赖(Milvus/BGE-M3 等),进程内子解释器运行。
"""
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


def test_mcnemar_paired_stats() -> None:
    """McNemar 精确二项检验 + 配对 bootstrap(替代双比例 z 的配对场景)。"""
    code = r"""
import importlib
h = importlib.import_module("2Milvus_RAG_Qa.RAG评测.eval_harness")

# 8 对全倒向一边 → p = 2*0.5^8 = 0.0078125(与 statsmodels 精确 McNemar 一致)
assert abs(h.mcnemar_pvalue(8, 0) - 0.0078125) < 1e-9
assert h.mcnemar_pvalue(0, 8) == h.mcnemar_pvalue(8, 0)   # 对称
assert h.mcnemar_pvalue(0, 0) == 1.0                       # 无不一致对
assert h.mcnemar_pvalue(5, 5) == 1.0
assert 0.05 < h.mcnemar_pvalue(1, 3) < 1.0                # 弱差异不显著

# 配对 bootstrap:同批 query 的 delta 均值与 CI 覆盖关系
low, mean, high = h.paired_bootstrap_delta_ci([0]*100 + [1]*100, [1]*100 + [0]*100)
assert -0.2 < mean < 0.2, (low, mean, high)
assert low <= mean <= high
print("MCNEMAR_OK")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "MCNEMAR_OK" in result.stdout


def test_jd_intent_routing() -> None:
    """意图路由:求职查询触发 JD;技术意图查询/无条件查询不触发。"""
    code = r"""
import importlib
d = importlib.import_module("2Milvus_RAG_Qa.core.dual_retrieval")

# 求职查询(信号词 + 可抽取条件)触发
assert d.should_trigger_jd("北京 java 15k 岗位") is True
assert d.should_trigger_jd("上海 python 20k 以上岗位") is True
assert d.should_trigger_jd("成都 C# 10k 岗位") is True
assert d.should_trigger_jd("java 岗位有哪些") is True      # 无条件但有方向
# 已分类的求职意图可补足信号词漏检
assert d.should_trigger_jd("帮我看看合适的工作", intent="就业薪资") is False  # 无条件

# 技术问题:无信号词 → 不触发
assert d.should_trigger_jd("什么是RAG") is False
assert d.should_trigger_jd("Python多线程怎么用") is False
# 含"面试"但带技术标记且无城市/薪资 → 技术问题,不触发(防 JD 污染)
assert d.should_trigger_jd("面试时会问哪些java原理") is False
assert d.should_trigger_jd("java 垃圾回收原理是什么") is False
# 有城市/薪资则视为求职,即使带技术标记
assert d.should_trigger_jd("北京 java 15k 岗位") is True
print("JD_ROUTING_OK")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "JD_ROUTING_OK" in result.stdout


def test_gate_score_isolation() -> None:
    """门控分数口径:JD 置顶标记(1.0)不参与门控,只看向量路重排分。"""
    code = r"""
import importlib
a = importlib.import_module("2Milvus_RAG_Qa.core.agent_graph")

class DummyRag:
    def _call_llm_with_retry(self, *args, **kwargs):
        return "充分"
    def _ask_llm(self, *args, **kwargs):
        return "改写的查询"
    def _generate(self, q, ctx):
        return "ok"

agent = a.AdaptiveRAG(DummyRag())
state0 = {"query": "q", "original_query": "q", "docs": [], "grade": "", "rewrite_count": 0, "answer": ""}

# 1. JD(1.0) 在前但向量 top1=0.3 ≤ 0.4 → 不能借 JD 置顶蒙混过关,直判不充分
s1 = dict(state0, docs=[
    {"text": "【招聘岗位】...", "score": 1.0, "source": "jd"},
    {"text": "doc", "score": 0.3, "source": "vector"},
])
r1 = agent._grade(s1)
assert r1["grade"] == "fail" and r1["grade_via"] == "score_gate", r1

# 2. 向量 top1=0.8 ≥ 0.7 → score_gate 直过
s2 = dict(state0, docs=[
    {"text": "【招聘岗位】...", "score": 1.0, "source": "jd"},
    {"text": "doc", "score": 0.8, "source": "vector"},
])
r2 = agent._grade(s2)
assert r2["grade"] == "pass" and r2["grade_via"] == "score_gate", r2

# 3. 仅 JD 命中(无向量文档) → jd_struct_only 直过
s3 = dict(state0, docs=[{"text": "【招聘岗位】...", "score": 1.0, "source": "jd"}])
r3 = agent._grade(s3)
assert r3["grade"] == "pass" and r3["grade_via"] == "jd_struct_only", r3

# 4. 空文档 → no_docs fail
r4 = agent._grade(state0)
assert r4["grade"] == "fail" and r4["grade_via"] == "no_docs", r4
print("GATE_ISOLATION_OK")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "GATE_ISOLATION_OK" in result.stdout


def test_context_assembly_tiers() -> None:
    """上下文组装三档规则:≤8 直塞 / 9-15 截断 top8 / >15 分层压缩(补空洞)。"""
    code = r"""
import importlib
import types
v = importlib.import_module("2Milvus_RAG_Qa.core.vector_store")

class FakeVS:
    FULL_PASSTHROUGH_DOCS = v.VectorStore.FULL_PASSTHROUGH_DOCS
    FORCE_COMPRESS_DOCS = v.VectorStore.FORCE_COMPRESS_DOCS
    FORCE_COMPRESS_CHARS = v.VectorStore.FORCE_COMPRESS_CHARS
    TRUNCATE_KEEP_DOCS = v.VectorStore.TRUNCATE_KEEP_DOCS

    def compress_contexts(self, query, contexts, max_chars=None):
        return ["L2_COMPRESSED"]

fake = FakeVS()

# 档1:5 文档且总字符 ≤ 12000 → 全文直塞
ctx5 = ["x" * 1000] * 5
assert v.VectorStore.build_layered_contexts(fake, "q", ctx5, max_chars=12000) == ctx5

# 档2:10 文档(9-15)且总字符 ≤ 24000 → 截断保留 top8
ctx10 = ["d%d" % i + "y" * 900 for i in range(10)]   # 总长 ~9k ≤ 12k
out2 = v.VectorStore.build_layered_contexts(fake, "q", ctx10, max_chars=12000)
assert len(out2) == 8 and out2 == ctx10[:8], len(out2)

# 档3:16 文档 → 分层压缩(top2 原文 + 压缩段)
ctx16 = ["d%d" % i + "y" * 900 for i in range(16)]
out3 = v.VectorStore.build_layered_contexts(fake, "q", ctx16, max_chars=12000)
assert out3 == ctx16[:2] + ["L2_COMPRESSED"], out3

# 档3b:文档少但总字符 > 24000 → 强制压缩;top2 已超预算时只保留 top2
ctx3 = ["z" * 9000] * 3
out3b = v.VectorStore.build_layered_contexts(fake, "q", ctx3, max_chars=12000)
assert out3b == ctx3[:2], len(out3b)
print("CONTEXT_TIERS_OK")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "CONTEXT_TIERS_OK" in result.stdout


def test_cache_intent_slot_isolation() -> None:
    """语义缓存一致性校验:意图/硬槽不一致不得复用缓存答案。"""
    code = r"""
import importlib
c = importlib.import_module("2Milvus_RAG_Qa.core.rag_cache")

# 意图不一致(求职 vs 技术)→ 拒绝
p1 = {"intent": "就业薪资", "slots": {"city": "北京"}, "answer": "a"}
assert c.RagCache._payload_consistent(p1, intent="技术问题", slots=None) is False
# 求职 vs 闲聊 → 拒绝(求职性判定不一致)
assert c.RagCache._payload_consistent(p1, intent="闲聊/其他", slots=None) is False
# 同求职类 + 硬槽一致 → 通过
assert c.RagCache._payload_consistent(p1, intent="就业薪资", slots={"city": "北京"}) is True
# 硬槽不一致(北京 vs 上海)→ 拒绝("近似 query 不同诉求"防误命中)
assert c.RagCache._payload_consistent(p1, intent="就业薪资", slots={"city": "上海"}) is False
# 薪资槽不一致 → 拒绝
p2 = {"intent": "就业薪资", "slots": {"city": "北京", "salary_min": 20}, "answer": "a"}
assert c.RagCache._payload_consistent(p2, intent="就业薪资", slots={"city": "北京", "salary_min": 15}) is False
# 技术类无槽 → 意图同为技术即通过
p3 = {"intent": "技术问题", "slots": {}, "answer": "a"}
assert c.RagCache._payload_consistent(p3, intent="技术问题", slots={}) is True
print("CACHE_ISOLATION_OK")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "CACHE_ISOLATION_OK" in result.stdout


def test_rewrite_merge_monotonic() -> None:
    """改写候选合并:跨轮并集去重、同文本取最高分、JD 槽位最前、统一重排截断。"""
    code = r"""
import importlib
d = importlib.import_module("2Milvus_RAG_Qa.core.dual_retrieval")

prev = [
    {"text": "jd1", "score": 1.0, "source": "jd"},
    {"text": "docA", "score": 0.9, "source": "vector"},
    {"text": "docB", "score": 0.6, "source": "vector"},
]
new = [
    {"text": "docA", "score": 0.5, "source": "vector"},   # 同文本低分 → 保留高分
    {"text": "docC", "score": 0.8, "source": "vector"},
]
out = d.merge_docs(prev, new, top_k=5)
texts = [x["text"] for x in out]
assert texts[0] == "jd1"                                  # JD 槽位最前
assert texts.count("docA") == 1                           # 按文本去重
assert out[1]["score"] == 0.9                             # 同文本保留最高分
assert "docC" in texts and "docB" in texts                # 并集
assert len([x for x in out if x["source"] == "vector"]) == 3

# JD 槽位上限
many_jd = [{"text": "j%d" % i, "score": 1.0, "source": "jd"} for i in range(6)]
out2 = d.merge_docs([], many_jd, top_k=5)
assert len([x for x in out2 if x["source"] == "jd"]) == d.JD_MAX_SLOTS
print("MERGE_MONOTONIC_OK")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "MERGE_MONOTONIC_OK" in result.stdout


def test_memory_user_isolation_guard() -> None:
    """记忆用户隔离:空 user_id 直接拒绝,绝不跨用户召回。"""
    code = r"""
import importlib
m = importlib.import_module("2Milvus_RAG_Qa.core.memory")

# 不执行 __init__(不连 MySQL),直接构造空壳实例
ml = m.MemoryLayer.__new__(m.MemoryLayer)
assert ml.search_facts("", "北京 java 岗位") == []
assert ml.search_facts(None, "北京 java 岗位") == []
assert ml._write_facts("", ["随便一条事实"]) == 0
print("MEMORY_ISOLATION_OK")
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "MEMORY_ISOLATION_OK" in result.stdout
