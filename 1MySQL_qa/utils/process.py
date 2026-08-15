"""文本预处理工具 — jieba 分词 + 停用词过滤。

使用方式:
    from utils.process import process_text
    tokens = process_text("什么是机器学习")
    # → ['什么', '机器', '学习']
"""

import jieba
from typing import List


# 中文停用词表（高频无意义词，过滤后提升检索精度）
_STOP_WORDS: set[str] = {
    "的", "了", "是", "在", "和", "有", "我", "你", "他", "她", "它",
    "就", "都", "也", "还", "要", "会", "能", "把", "被", "让", "给",
    "对", "从", "到", "与", "或", "但", "而", "且", "所", "为", "以",
    "及", "之", "其", "等", "这", "那", "哪", "吗", "呢", "吧", "啊",
    "哦", "嗯", "哈", "嘛", "哟", "嘛", "呀", "呗", "啦", "哇",
}


def process_text(text: str) -> List[str]:
    """对中文文本进行 jieba 分词，过滤停用词和空白字符。

    Args:
        text: 待分词的原始文本

    Returns:
        分词后的词语列表；输入为空或非字符串时返回空列表
    """
    if not text or not isinstance(text, str):
        return []

    # 精确模式分词
    raw_words: List[str] = jieba.lcut(text)

    # 过滤停用词和空白字符
    return [
        word for word in raw_words
        if word not in _STOP_WORDS and word.strip()
    ]
