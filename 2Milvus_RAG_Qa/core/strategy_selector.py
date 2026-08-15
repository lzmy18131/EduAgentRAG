"""检索策略选择器 — 根据查询特征选择最优的 RAG 检索策略。

四种策略：
    1. 直接检索 — 查询意图明确，直接向量检索
    2. 假设问题检索（HyDE）— 查询抽象，先让 LLM 生成假答案再检索
    3. 子问题检索 — 复杂多实体查询，拆分子问题分别检索合并
    4. 回溯问题检索 — 复杂问题简化为基础问题再检索
"""

import json
import re
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI

from base.config import cfg
from base.logger import logger


# ── 意图 → 检索策略 规则映射(规则优先,未命中回退 LLM)──
# 2026-08-13 消融实测(旧语料/96 条金标):HyDE 在本语料上 HitRate@5 0.7604→0.5208 负收益,故全部路由到直接检索;
# HyDE 保留为可选策略(仅 LLM 选择器可显式启用),不作为默认规则
INTENT_TO_STRATEGY = {
    "课程咨询": "直接检索", "就业薪资": "直接检索", "技术问题": "直接检索",
    "学习方法": "直接检索", "概念解释": "直接检索",
    "项目实战": "子问题检索", "职业规划": "子问题检索",
    "工具安装": "直接检索", "报名流程": "直接检索",
}


class StrategySelector:
    """使用 LLM 选择最优检索策略。"""

    STRATEGIES = ["直接检索", "假设问题检索", "子问题检索", "回溯问题检索"]

    def __init__(self) -> None:
        self._client = OpenAI(api_key=cfg.LLM_API_KEY, base_url=cfg.LLM_BASE_URL, timeout=120)
        self._model = cfg.LLM_MODEL

    @staticmethod
    def rule_based_select(intent: str | None) -> str | None:
        """根据意图返回规则映射的检索策略。

        Args:
            intent: 意图类别标签

        Returns:
            命中的策略字符串；意图不在映射表（含 None / 空串）返回 None，
            由调用方回退到 LLM select_strategy。
        """
        if not intent:
            return None
        return INTENT_TO_STRATEGY.get(intent)

    def select_strategy(self, query: str) -> str:
        """调用 LLM 选择最合适的检索策略。

        Args:
            query: 用户查询

        Returns:
            "直接检索" | "假设问题检索" | "子问题检索" | "回溯问题检索"
        """
        prompt = self._get_strategy_prompt(query)
        strategy = self._parse_strategy(self._call_llm(prompt))

        # 兜底：如果 LLM 返回不认识的策略
        if strategy not in self.STRATEGIES:
            logger.warning("LLM 返回未知策略 [%s]，降级为直接检索", strategy)
            strategy = "直接检索"

        logger.info("策略选择: query=%s → %s", query[:40], strategy)
        return strategy

    def _call_llm(self, prompt: str) -> str:
        """策略选择 LLM 调用：重试 3 次 + 空 content 视为失败。

        reasoning_effort 统一传 "low"（deepseek-v4-pro 不传会 100% 空返回，实测验证）。
        max_tokens 提到 128：32 会被 reasoning 吃满导致 content 空返回（实测验证）。
        """
        last_err: Exception | None = None
        for attempt in range(1, 4):  # deepseek 偶发空返回，重试 3 次
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=128,
                    # deepseek-v4-pro 实测：不传 reasoning_effort / thinking disabled
                    # 时 100% 空返回 content，故统一传 low
                    reasoning_effort="low",
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                last_err = ValueError("LLM 策略选择返回空 content")
            except Exception as e:
                last_err = e
            if attempt < 3:
                time.sleep(2 * attempt)
        logger.error("策略选择 API 调用失败（重试 3 次）: %s，降级为直接检索", last_err)
        return "直接检索"

    @staticmethod
    def _parse_strategy(raw: str) -> str:
        """从 JSON 或纯文本中提取策略名(结构化输出 + 文本回退)。"""
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                v = json.loads(m.group(0)).get("strategy", "")
                if v:
                    return str(v).strip()
            except Exception:
                pass
        return raw.strip()

    def _get_strategy_prompt(self, query: str) -> str:
        """生成策略选择提示词。"""
        return f"""你是检索策略专家，请分析用户查询并选择最优策略。

{query}

1. 直接检索：查询意图明确、关键词清晰，不需要改写
   示例："AI课程学费多少"、"Python多线程怎么用"

2. 假设问题检索（HyDE）：查询较抽象、模糊，直接检索效果差
   示例："人工智能在教育的应用有哪些"、"如何提升编程能力"

3. 子问题检索：涉及多个实体或需比较，需拆开分别检索
   示例："比较Milvus和Zilliz的优缺点"、"Java和Python分别适合什么场景"

4. 回溯问题检索：查询过于复杂，需简化为更基础的表述
   示例："有100亿数据想存Milvus怎么做" → "Milvus大数据量存储方案"

请**只输出** JSON 对象,格式为:{{"strategy": "直接检索"}} 或 {{"strategy": "假设问题检索"}} 或 {{"strategy": "子问题检索"}} 或 {{"strategy": "回溯问题检索"}}"""
