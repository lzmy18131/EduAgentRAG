"""Edu 文档加载器 — 中文文本递归切分器。

专为中文教育文档优化：按段落、句子、分号、逗号逐级切分，
保证语义完整性，避免在词语中间断开。
"""

import re
from typing import List


class ChineseRecursiveTextSplitter:
    """中文递归文本切分器。

    切分优先级（从大到小）：
        段落(\n\n) → 句子(。！？) → 分句(；) → 子句(，、)
    确保每次切分都在自然语义边界，不截断词语。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 切分分隔符优先级
        self._separators = [
            r"\n\n",           # 段落
            r"\n",             # 换行
            r"[。！？!?]",     # 句子结束
            r"[；;]",          # 分句
            r"[，,、]",        # 子句
            r"\s+",            # 空白
        ]

    def split_text(self, text: str) -> List[str]:
        """将文本递归切分为 chunks。

        Args:
            text: 待切分的原始文本

        Returns:
            切分后的文本块列表
        """
        if not text or not text.strip():
            return []

        chunks = self._recursive_split(text.strip())
        return self._merge_chunks(chunks)

    def _recursive_split(self, text: str) -> List[str]:
        """递归切分：尝试每个分隔符，直到 chunks 足够小。"""
        if len(text) <= self.chunk_size:
            return [text]

        for sep in self._separators:
            parts = re.split(f"({sep})", text)
            # 合并分隔符到前一部分
            merged = []
            i = 0
            while i < len(parts):
                if i + 1 < len(parts) and re.match(sep, parts[i + 1]):
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1

            # 如果切出了多个部分
            if len(merged) > 1:
                result = []
                for part in merged:
                    if part.strip():
                        result.extend(self._recursive_split(part.strip()))
                return result

        # 所有分隔符都切不动，强制按长度切
        return self._force_split(text)

    def _force_split(self, text: str) -> List[str]:
        """强制按字符数切分（兜底策略）。"""
        return [
            text[i:i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """合并过短的 chunks，确保每个 chunk 接近 chunk_size。"""
        if not chunks:
            return []

        merged: List[str] = []
        current = ""

        for chunk in chunks:
            if len(current) + len(chunk) <= self.chunk_size:
                current += chunk
            else:
                if current:
                    merged.append(current)
                current = chunk

        if current:
            merged.append(current)

        return merged


def split_documents(docs: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """对文档列表进行切分，保留原始元数据。

    Args:
        docs: 文档列表，每个元素为 {"text": str, "metadata": dict}
        chunk_size: 切分大小
        chunk_overlap: 重叠大小

    Returns:
        切分后的文档列表
    """
    splitter = ChineseRecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    result = []
    for doc in docs:
        texts = splitter.split_text(doc["text"])
        for i, text in enumerate(texts):
            result.append({
                "text": text,
                "metadata": {**doc.get("metadata", {}), "chunk_index": i},
            })
    return result
