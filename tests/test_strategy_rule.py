"""StrategySelector 意图→策略规则映射单元测试。"""

import importlib
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_strategy_module = importlib.import_module("2Milvus_RAG_Qa.core.strategy_selector")
INTENT_TO_STRATEGY = _strategy_module.INTENT_TO_STRATEGY
StrategySelector = _strategy_module.StrategySelector


def test_intent_to_strategy_mapping_complete() -> None:
    """映射表覆盖全部 9 个意图,且值符合契约。

    2026-08-13 更新(旧语料/96 条金标):HyDE 消融实测负收益(HitRate@5 0.7604→0.5208),
    "学习方法/概念解释" 由假设问题检索改为直接检索。
    """
    expected = {
        "课程咨询": "直接检索", "就业薪资": "直接检索", "技术问题": "直接检索",
        "学习方法": "直接检索", "概念解释": "直接检索",
        "项目实战": "子问题检索", "职业规划": "子问题检索",
        "工具安装": "直接检索", "报名流程": "直接检索",
    }
    assert INTENT_TO_STRATEGY == expected


def test_rule_based_select_known_intents() -> None:
    """已知意图返回对应策略。"""
    assert StrategySelector.rule_based_select("技术问题") == "直接检索"
    assert StrategySelector.rule_based_select("学习方法") == "直接检索"
    assert StrategySelector.rule_based_select("项目实战") == "子问题检索"
    assert StrategySelector.rule_based_select("报名流程") == "直接检索"


def test_rule_based_select_unknown_or_none() -> None:
    """未知意图 / None / 空串均返回 None，走 LLM 兜底。"""
    assert StrategySelector.rule_based_select("知识检索") is None
    assert StrategySelector.rule_based_select(None) is None
    assert StrategySelector.rule_based_select("") is None
