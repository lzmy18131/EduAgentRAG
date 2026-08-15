"""FaqSemanticSearch 离线单元测试 — 不加载真实模型、不依赖 MySQL/Redis。

直接注入内存矩阵与编码函数，覆盖阈值 / 分差 / 版本号缓存 key 逻辑。
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_faq_module = importlib.import_module("1MySQL_qa.retrieval.faq_semantic")
FaqSemanticSearch = _faq_module.FaqSemanticSearch
process_text = _faq_module.process_text

_QUESTIONS = [
    "课程咨询的问题",
    "就业薪资的问题",
    "学习方法的问题",
    "项目实战的问题",
    "技术问题咨询",
]
_ANSWERS = {q: f"标准答案-{i}" for i, q in enumerate(_QUESTIONS)}


class FakeRedis:
    """内存版 Redis，模拟版本号缓存读写。"""

    def __init__(self) -> None:
        self.available = True
        self.store: dict[str, str] = {}
        self.version = 3

    def get_faq_version(self) -> int:
        return self.version

    def get_faq_answer(self, norm_key: str, version: int) -> str | None:
        return self.store.get(f"faq:v{version}:ans:{norm_key}")

    def set_faq_answer(self, norm_key: str, version: int, answer: str, ttl: int = 86400) -> bool:
        self.store[f"faq:v{version}:ans:{norm_key}"] = answer
        return True


class FakeMysql:
    """内存版 MySQL，按问题回源答案。"""

    def __init__(self, questions: list[str], answers: dict[str, str]) -> None:
        self._questions = questions
        self._answers = answers

    def fetch_questions(self) -> list[str]:
        return list(self._questions)

    def fetch_answer(self, question: str) -> str | None:
        return self._answers.get(question)


@pytest.fixture
def make_searcher(monkeypatch):
    """构造不加载真实模型的 FaqSemanticSearch，返回 (searcher, redis)。"""

    def _make(matrix: np.ndarray, encode_fn):
        # 阻止真实模型下载与真实数据加载
        monkeypatch.setattr(
            FaqSemanticSearch, "_load_model",
            lambda self: setattr(self, "_model", None),
        )
        monkeypatch.setattr(FaqSemanticSearch, "_load_data", lambda self: None)

        redis = FakeRedis()
        mysql = FakeMysql(_QUESTIONS, _ANSWERS)
        searcher = FaqSemanticSearch(redis, mysql)

        # 注入内存矩阵与编码函数
        searcher._model = object()
        searcher.questions = list(_QUESTIONS)
        searcher.matrix = np.asarray(matrix, dtype=np.float32)
        searcher._encode = encode_fn
        return searcher, redis

    return _make


def test_search_below_threshold_returns_none(make_searcher) -> None:
    """top1 低于阈值 → 未命中降级 RAG。"""
    searcher, _ = make_searcher(
        np.eye(len(_QUESTIONS)),
        lambda text: np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    answer, msg = searcher.search("某个问题", threshold=0.92)
    assert answer is None
    assert "阈值" in msg
    assert "未命中降级 RAG" in msg


def test_search_small_margin_returns_none(make_searcher) -> None:
    """top1 达标但非高分(低于 0.93)且 top1-top2 分差过小 → 视为歧义放行降级。"""
    searcher, _ = make_searcher(
        np.eye(len(_QUESTIONS)),
        lambda text: np.array([0.92, 0.90, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    answer, msg = searcher.search("某个问题", threshold=0.92)
    assert answer is None
    assert "分差" in msg
    assert "未命中降级 RAG" in msg


def test_search_high_confidence_skips_margin(make_searcher) -> None:
    """top1 高分(≥0.93)时跳过 margin 判定，即使 top1/top2 分差过小也直接命中。

    场景："什么是过拟合" vs "什么是过拟合?" 这类近似重复原问题，
    高分并列(分差 0.02)不应被判为语义歧义。
    """
    searcher, _ = make_searcher(
        np.eye(len(_QUESTIONS)),
        lambda text: np.array([0.95, 0.93, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    answer, msg = searcher.search("某个问题", threshold=0.92)
    assert answer == "标准答案-0"
    assert "匹配成功" in msg


def test_search_success_writes_version_cache(make_searcher) -> None:
    """命中成功：回源答案并写入版本号缓存。"""
    searcher, redis = make_searcher(
        np.eye(len(_QUESTIONS)),
        lambda text: np.array([0.99, 0.5, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    query = "如何报名课程"
    answer, msg = searcher.search(query, threshold=0.92)
    assert answer == "标准答案-0"
    assert "匹配成功" in msg

    norm_key = "_".join(process_text(query))
    version = redis.get_faq_version()
    assert redis.store[f"faq:v{version}:ans:{norm_key}"] == "标准答案-0"


def test_search_paraphrase_maps_back_to_original(make_searcher) -> None:
    """命中同义改写时，通过改写→原问题映射回源标准答案。"""
    searcher, _ = make_searcher(
        np.eye(len(_QUESTIONS)),
        lambda text: np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    # 模拟扩充后的语料库：索引 1 为改写，回源到原问题 _QUESTIONS[0]
    searcher.questions = [
        "课程咨询的问题",
        "课程咨询的问题的口语改写",
        "就业薪资的问题",
        "学习方法的问题",
        "项目实战的问题",
    ]
    searcher._answer_map = [
        "课程咨询的问题",
        "课程咨询的问题",
        "就业薪资的问题",
        "学习方法的问题",
        "项目实战的问题",
    ]
    answer, msg = searcher.search("如何报名课程", threshold=0.92)
    assert answer == "标准答案-0"  # 命中改写(索引1)，回源到原问题(索引0)
    assert "匹配成功" in msg


def test_search_cache_hit_skips_encode(make_searcher) -> None:
    """版本号缓存命中直接返回，不触发编码。"""
    searcher, redis = make_searcher(
        np.eye(len(_QUESTIONS)),
        lambda text: np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    query = "如何报名课程"
    norm_key = "_".join(process_text(query))
    version = redis.get_faq_version()
    redis.store[f"faq:v{version}:ans:{norm_key}"] = "缓存答案"

    calls = {"n": 0}

    def encode_fn(text: str) -> np.ndarray:
        calls["n"] += 1
        return np.zeros(len(_QUESTIONS), dtype=np.float32)

    searcher._encode = encode_fn
    answer, msg = searcher.search(query, threshold=0.92)
    assert answer == "缓存答案"
    assert msg == "缓存命中"
    assert calls["n"] == 0


def test_search_model_unavailable_degrades_gracefully(make_searcher) -> None:
    """模型不可用时返回 None，不抛异常。"""
    searcher, _ = make_searcher(
        np.eye(len(_QUESTIONS)),
        lambda text: np.zeros(len(_QUESTIONS), dtype=np.float32),
    )
    searcher._model = None
    answer, msg = searcher.search("某个问题", threshold=0.92)
    assert answer is None
    assert "模型" in msg
