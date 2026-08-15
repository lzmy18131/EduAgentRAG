# -*- coding: utf-8 -*-
"""D2 数据治理:语料审计(去重/清洗/代码块/语言统计) + 代码块分块保护。

审计对象:线上 Milvus 知识库(358,221 行,真实语料,2026-08-15 重建)。
  - 精确重复 chunk(md5)统计
  - 过短/空白 chunk 统计
  - 代码块占比(代码围栏/缩进代码)
  - 语言分布(中/英/日/韩/葡等,启发式字符区间)
  - chunk 长度分布
  输出 data_governance_report.json。

代码块单独处理:
  markdown 的 ``` 围栏代码块在分块前用占位符保护,分块后还原——
  避免切分器在代码中间截断,保证代码块作为整体进入子块。
"""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from base.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPORT_PATH = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "data_governance_report.json"

# ──────────────── 语言启发式检测 ────────────────
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
# 葡语特征字符(语料含 TensorFlow.js 葡语文档)
_PT_RE = re.compile(r"[ãõçêôáéíóúà]")

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_INDENT_CODE_RE = re.compile(r"(?m)^ {4}\S|\t\S")


def detect_language(text: str) -> str:
    """启发式语言判定:中日韩优先,拉丁语系带葡语特征字符判 pt,其余 en。"""
    n = len(text) or 1
    cjk = len(_CJK_RE.findall(text)) / n
    hangul = len(_HANGUL_RE.findall(text)) / n
    kana = len(_KANA_RE.findall(text)) / n
    latin = len(_LATIN_RE.findall(text)) / n
    pt_marks = len(_PT_RE.findall(text))
    if cjk > 0.1:
        return "zh"
    if hangul > 0.1:
        return "ko"
    if kana > 0.1:
        return "ja"
    if pt_marks >= 2 and latin > 0.2:
        return "pt"
    if latin > 0.3:
        return "en"
    return "other"


# ──────────────── 代码块保护(分块前占位,分块后还原) ────────────────

def protect_code_blocks(text: str) -> tuple[str, dict[str, str]]:
    """把 ``` 围栏代码块替换为占位符,返回 (处理文本, 占位符映射)。"""
    blocks: dict[str, str] = {}

    def _sub(m: re.Match) -> str:
        key = f"__CODEBLOCK_{len(blocks):04d}__"
        blocks[key] = m.group(0)
        return key

    return _CODE_FENCE_RE.sub(_sub, text), blocks


def restore_code_blocks(text: str, blocks: dict[str, str]) -> str:
    """把占位符还原为代码块。"""
    for key, code in blocks.items():
        text = text.replace(key, code)
    return text


def split_with_code_units(text: str, splitter) -> list[str]:
    """子块切分:代码块作为不可分割单元,非代码部分用 splitter 正常切。

    Args:
        text:    父块文本
        splitter: langchain 文本切分器

    Returns:
        子块列表,其中每个 ``` 围栏代码块整体保留(不截断)。
    """
    parts = re.split(r"(\x60\x60\x60[\s\S]*?\x60\x60\x60)", text)
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("```"):
            out.append(part)
        else:
            out.extend(splitter.split_text(part))
    return out


# ──────────────── 审计 ────────────────

def audit_milvus_corpus(
    collection: str,
    sample_per_window: int = 1000,
    window_step: int = 8000,
    max_samples: int = 30000,
) -> dict:
    """抽样审计 Milvus 语料,返回统计 dict(真实数据)。"""
    from pymilvus import MilvusClient
    from base.config import cfg as _cfg

    client = MilvusClient(uri=f"http://{_cfg.MILVUS_HOST}:{_cfg.MILVUS_PORT}")
    client.use_database(_cfg.MILVUS_DB_NAME)
    total = client.get_collection_stats(collection).get("row_count", 0)

    seen_hash: set[str] = set()
    dup = short = code = 0
    langs: dict[str, int] = {}
    lengths: list[int] = []
    sampled = 0

    offset = 0
    while offset < total and sampled < max_samples:
        try:
            rows = client.query(
                collection_name=collection,
                filter='chunk_type == "child"',
                output_fields=["text"],
                limit=sample_per_window,
                offset=offset,
            )
        except Exception:
            # offset+limit 窗口越界时收敛采样步长
            offset += window_step
            continue
        for r in rows:
            text = r.get("text") or ""
            sampled += 1
            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if h in seen_hash:
                dup += 1
            seen_hash.add(h)
            if len(text.strip()) < 20:
                short += 1
            if _CODE_FENCE_RE.search(text) or _INDENT_CODE_RE.search(text):
                code += 1
            lang = detect_language(text)
            langs[lang] = langs.get(lang, 0) + 1
            lengths.append(len(text))
        offset += window_step

    lengths.sort()
    n = sampled or 1
    report = {
        "collection": collection,
        "total_rows": total,
        "sampled": sampled,
        "exact_duplicate_chunks": dup,
        "duplicate_rate": round(dup / n, 4),
        "short_chunks_lt20": short,
        "short_rate": round(short / n, 4),
        "code_block_chunks": code,
        "code_rate": round(code / n, 4),
        "language_distribution": {
            k: {"count": v, "rate": round(v / n, 4)} for k, v in sorted(langs.items())
        },
        "chunk_len_p50_p90_p99": [lengths[n // 2], lengths[int(n * 0.9)], lengths[int(n * 0.99)]],
        "audit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("语料审计完成: 抽样 %d/%d, 重复率 %.2f%%", sampled, total, report["duplicate_rate"] * 100)
    return report


def clean_whitespace(text: str) -> str:
    """文本清洗:统一换行、去行首尾空白、压缩 3+ 空行。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def main() -> None:
    """CLI:语料审计(默认审计 cfg 配置的知识库 collection)。"""
    import argparse

    from base.config import cfg as _cfg

    parser = argparse.ArgumentParser(description="D2 语料审计(去重/清洗/代码块/语言)")
    parser.add_argument("--collection", default=_cfg.MILVUS_COLLECTION)
    parser.add_argument("--max-samples", type=int, default=30000)
    args = parser.parse_args()

    report = audit_milvus_corpus(
        args.collection, sample_per_window=1000, window_step=1000,
        max_samples=args.max_samples,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
