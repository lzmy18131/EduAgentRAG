"""CC 包 2（P1+P2 优化）回归测试。

分两部分：
  1. 源码断言：验证契约要求的符号/参数已落地（不加载重依赖，快）
  2. 子进程逻辑测试：加载真实环境，验证纯函数/缓存/压缩/MMR 逻辑正确
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
        timeout=180,
        check=False,
    )


# ──────────────── 源码断言（不加载重依赖）───────────────

def test_prompt_numbered_citations_and_coreference() -> None:
    """D9/D11：RAG prompt 要求 [编号] 引用；新增指代消解 prompt。"""
    source = (PROJECT_ROOT / "2Milvus_RAG_Qa" / "core" / "prompts.py").read_text(encoding="utf-8")
    assert "编号" in source
    assert "[1]" in source
    assert "coreference_prompt" in source
    assert "{history}" in source


def test_vector_store_hnsw_and_compression_mmr() -> None:
    """9b/E14/D10/E15：HNSW 索引参数、MMR、压缩、filter 提取均落地。"""
    source = (PROJECT_ROOT / "2Milvus_RAG_Qa" / "core" / "vector_store.py").read_text(encoding="utf-8")
    assert '"HNSW"' in source
    assert "efConstruction" in source
    assert "_HNSW_M = 16" in source
    assert "_HNSW_EF_CONSTRUCTION = 200" in source
    assert "_HNSW_EF_SEARCH = 128" in source
    # S6 向量量化:保留 IVF_PQ 切换能力(默认 HNSW,rebuild_dense_index 实测对比)
    assert "IVF_PQ" in source
    assert "_IVF_NPROBE" in source
    assert "rebuild_dense_index" in source
    assert "_MMR_LAMBDA = 0.7" in source
    assert "compress_contexts" in source
    assert "_build_filter_expr" in source
    assert "hybrid_search_with_rerank_scored" in source


def test_rag_system_reject_cache_stream_multiturn() -> None:
    """E13/D12/E16/D11：拒答阈值、语义缓存、流式、多轮符号落地。"""
    source = (PROJECT_ROOT / "2Milvus_RAG_Qa" / "core" / "rag_system.py").read_text(encoding="utf-8")
    assert "REJECT_THRESHOLD: float = 0.15" in source
    assert "知识库中未找到相关资料" in source
    assert "session_id" in source
    assert "query_stream" in source
    assert "_stream_llm" in source
    assert "build_layered_contexts" in source  # 上下文分级组装(替代压缩直调)
    assert "_resolve_coreference" in source
    assert "get_semantic" in source


def test_app_stream_and_sources_ui() -> None:
    """E16/D9/D11：/chat/stream 端点、sources 可点击块、session_id 前端落地。"""
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "session_id" in source
    assert "/chat/stream" in source
    assert "StreamingResponse" in source
    assert "text/event-stream" in source
    assert "source-item" in source
    assert "source-head" in source


def test_rag_cache_threshold_and_methods() -> None:
    """D12/D11：语义缓存阈值 0.95、会话历史方法落地。"""
    source = (PROJECT_ROOT / "2Milvus_RAG_Qa" / "core" / "rag_cache.py").read_text(encoding="utf-8")
    assert "SEMANTIC_SIMILARITY_THRESHOLD = 0.95" in source
    assert "SEMANTIC_CACHE_TTL = 24 * 3600" in source
    assert "append_history" in source
    assert "get_semantic" in source
    assert "set_semantic" in source


def test_config_compress_switch() -> None:
    """D10：压缩开关与字符上限进 config。"""
    ini = (PROJECT_ROOT / "static" / "config.ini").read_text(encoding="utf-8")
    assert "context_compress_enabled" in ini
    assert "context_max_chars" in ini
    cfg_source = (PROJECT_ROOT / "base" / "config.py").read_text(encoding="utf-8")
    assert "CONTEXT_COMPRESS_ENABLED" in cfg_source
    assert "CONTEXT_MAX_CHARS" in cfg_source


# ──────────────── 子进程逻辑测试（加载真实环境）───────────────

def test_cc_package2_core_logic() -> None:
    """纯函数/缓存/压缩/MMR 逻辑在真实环境下计算正确。"""
    code = r"""
import importlib
import numpy as np

vs_mod = importlib.import_module("2Milvus_RAG_Qa.core.vector_store")
VectorStore = vs_mod.VectorStore

# 1. E15：source_filter 过滤表达式
assert VectorStore._build_filter_expr(None) == 'chunk_type == "child"'
assert VectorStore._build_filter_expr("java") == 'chunk_type == "child" and source == "java"'
assert VectorStore._build_filter_expr('a"b') == 'chunk_type == "child" and source == "a\\"b"'

# 2. D10：句子切分
assert VectorStore._split_sentences("你好。世界！测试") == ["你好。", "世界！", "测试"]

# 3. E14：MMR 多样性选择（跳过与已选高度相似的候选项）
scores = [0.9, 0.89, 0.88]
sim = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
idx = VectorStore._mmr_select(scores, sim, 0.7, 2)
assert idx == [0, 2], idx

# 4. D10：上下文压缩（fake embedding，高相关句保留且受字符上限约束）
cfg = importlib.import_module("base.config").cfg
cfg.CONTEXT_COMPRESS_ENABLED = True

class FakeEF:
    def __call__(self, texts):
        dense = []
        for t in texts:
            if "高相关" in t or t == "query":
                dense.append([1.0, 0.0, 0.0, 0.0])
            else:
                dense.append([0.0, 1.0, 0.0, 0.0])
        return {"dense": dense, "sparse": []}

vs = VectorStore.__new__(VectorStore)
vs._ef = FakeEF()
contexts = ["高相关句。" + "低相关句。" * 50]
compressed = vs.compress_contexts("query", contexts, max_chars=30)
assert compressed, "压缩结果不应为空"
total = sum(len(c) for c in compressed)
assert total <= 40, total
assert "高相关句" in "".join(compressed)

# 5. D12/D11：RagCache 语义缓存 + 会话历史（fake redis）
rag_cache = importlib.import_module("2Milvus_RAG_Qa.core.rag_cache")
RagCache = rag_cache.RagCache

class FakeRedis:
    def __init__(self):
        self.store = {}
    def ping(self):
        return True
    def get(self, k):
        return self.store.get(k)
    def setex(self, k, ttl, v):
        self.store[k] = v
        return True
    def set(self, k, v):
        self.store[k] = v
    def scan_iter(self, match=None, count=100):
        prefix = (match or "").replace("*", "")
        for k in list(self.store.keys()):
            if k.startswith(prefix):
                yield k

rc = RagCache.__new__(RagCache)
rc._conn = FakeRedis()
rc.available = True

q_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
rc.set_semantic("什么是RAG", q_emb, {
    "answer": "RAG是检索增强生成", "sources": [], "intent": "技术问题", "strategy": "直接检索",
})
p1 = rc.get_semantic("什么是RAG", q_emb, threshold=0.95)
assert p1 is not None and p1["answer"] == "RAG是检索增强生成"
# 语义最近邻命中（不同文本，embedding 高度相似）
q2 = np.array([0.99, 0.01, 0.0], dtype=np.float32)
p2 = rc.get_semantic("啥是RAG", q2, threshold=0.95)
assert p2 is not None and p2["answer"] == "RAG是检索增强生成"
# 语义不相似 → 未命中
q3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
assert rc.get_semantic("今天天气", q3, threshold=0.95) is None

rc.append_history("s1", "什么是RAG", "RAG是检索增强生成")
h = rc.get_history("s1")
assert len(h) == 2
assert h[0] == {"role": "user", "content": "什么是RAG"}
assert h[1] == {"role": "assistant", "content": "RAG是检索增强生成"}
for i in range(12):
    rc.append_history("s1", f"q{i}", f"a{i}")
h = rc.get_history("s1")
assert len(h) <= 20  # 10 轮上限 * 2 条

# 6. D9/D11：prompt 模板内容
prompts_mod = importlib.import_module("2Milvus_RAG_Qa.core.prompts")
RAGPrompts = prompts_mod.RAGPrompts
rag_p = RAGPrompts.rag_answer_prompt()
assert "[1]" in rag_p and "编号" in rag_p
coref = RAGPrompts.coreference_prompt()
assert "{history}" in coref and "{question}" in coref

print("CC_PKG2_OK")
"""
    result = run_project_python(code=code)
    assert result.returncode == 0, result.stderr
    assert "CC_PKG2_OK" in result.stdout


def test_rag_system_query_stream_is_generator() -> None:
    """E16：RAGSystem 具备 query_stream 流式入口（可导入断言）。"""
    code = r"""
import importlib
import inspect
mod = importlib.import_module("2Milvus_RAG_Qa.core.rag_system")
RAGSystem = mod.RAGSystem
assert hasattr(RAGSystem, "query_stream")
assert "session_id" in inspect.signature(RAGSystem.query).parameters
assert "session_id" in inspect.signature(RAGSystem.query_stream).parameters
print("STREAM_OK")
"""
    result = run_project_python(code=code)
    assert result.returncode == 0, result.stderr
    assert "STREAM_OK" in result.stdout
