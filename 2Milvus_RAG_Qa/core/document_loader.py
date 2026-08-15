"""文档加载与父子分块模块 — 支持多种格式文档加载 + 父子块切分。

文档加载：
    - .md     → Markdown 专用切分
    - .txt    → 通用递归字符切分
    - .pdf    → PyMuPDF 加载后切分

时间不够可以先用 .md/.txt，PDF 加载需额外安装 pymupdf。
"""

import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ── sys.path ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from base.config import cfg
from base.logger import logger
from ..edu_document_loaders.ocr_loader import ocr_extractor, _IMAGE_EXTS
from .data_governance import (
    protect_code_blocks,
    restore_code_blocks,
    clean_whitespace,
    split_with_code_units,
)


# ── 文件后缀 → 加载方式映射 ──
_TEXT_EXTS = {".txt", ".py", ".java", ".sql", ".json", ".yaml", ".xml", ".csv"}
_MD_EXTS = {".md", ".markdown"}
_DOC_EXTS = {".pdf", ".docx", ".doc"}


def load_documents_from_directory(data_dir: str) -> list[dict[str, Any]]:
    """从目录加载所有支持格式的文档，附加元数据。

    Args:
        data_dir: 数据目录路径

    Returns:
        [{"text": "...", "metadata": {"source": "ai", "file_path": "...", ...}}, ...]
    """
    documents: list[dict] = []
    data_path = Path(data_dir)
    if not data_path.is_dir():
        logger.warning("数据目录不存在: %s", data_dir)
        return documents

    # 从目录名提取学科/主题
    subject = data_path.name.replace("_data", "").upper()

    converter = None

    for file_path in sorted(data_path.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        text: str | None = None

        try:
            # ── markdown / txt ──
            if suffix in _TEXT_EXTS | _MD_EXTS:
                text = file_path.read_text(encoding="utf-8", errors="ignore")

            # ── pdf / docx（docling 万能转换）──
            elif suffix in _DOC_EXTS:
                if converter is None:
                    from docling.document_converter import DocumentConverter
                    converter = DocumentConverter()
                result = converter.convert(str(file_path))
                text = result.document.export_to_markdown()  # 转 Markdown 保留结构信息
                # 扫描型 PDF：docling 提不出文字 → PaddleOCR 兜底
                if not text or len(text.strip()) < 20:
                    logger.info("PDF 无文字层 [%s]，降级 PaddleOCR 逐页识别", file_path.name)
                    text = ocr_extractor.extract_from_pdf(file_path)

            # ── 图片（png/jpg 等）→ PaddleOCR ──
            elif suffix in _IMAGE_EXTS:
                logger.info("图片文档 [%s]，PaddleOCR 识别", file_path.name)
                text = ocr_extractor.extract_from_image(file_path)

            else:
                continue
        except Exception as e:
            logger.warning("文件加载失败 [%s]: %s", file_path.name, e)
            continue

        if not text or not text.strip():
            continue

        documents.append({
            "text": text.strip(),
            "metadata": {
                "source": subject,
                "file_path": str(file_path.resolve()),
                "relative_path": file_path.relative_to(data_path).as_posix(),
                "file_name": file_path.name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        })

    logger.info("文档加载完成：%d 篇，目录=%s", len(documents), data_dir)
    return documents


def process_documents(
    data_dir: str | None = None,
    parent_size: int | None = None,
    child_size: int | None = None,
    overlap: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """加载文档并执行父子分块。

    Args:
        data_dir:     数据目录路径
        parent_size:  父块大小（token 数），默认取 config
        child_size:   子块大小，默认取 config
        overlap:      重叠大小，默认取 config

    Returns:
        (parent_chunks, child_chunks) 两个列表
    """
    parent_size = cfg.PARENT_CHUNK_SIZE if parent_size is None else parent_size
    child_size = cfg.CHILD_CHUNK_SIZE if child_size is None else child_size
    overlap = cfg.CHUNK_OVERLAP if overlap is None else overlap
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size 和 child_size 必须大于 0")
    if overlap < 0 or overlap >= min(parent_size, child_size):
        raise ValueError("overlap 必须小于 parent_size 和 child_size")

    if data_dir is None:
        data_dir = str(_PROJECT_ROOT / "2Milvus_RAG_Qa" / "data" / "ai_data")

    documents = load_documents_from_directory(data_dir)
    if not documents:
        return [], []

    # ── 初始化切分器 ──
    # 通用递归切分器（用于 txt / PDF / docx）
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", ".", "！", "？", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", ".", "！", "？", " ", ""],
    )
    # Markdown 专用切分器（按标题层级切）
    md_headers = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_headers)

    parent_chunks: list[dict] = []
    child_chunks: list[dict] = []
    deduped_parents = 0
    deduped_children = 0

    for doc in documents:
        # D2 数据治理:统一清洗空白(换行规范化/去行首尾空白/压缩空行)
        text = clean_whitespace(doc["text"])
        meta = doc["metadata"]
        is_md = Path(meta["file_path"]).suffix.lower() == ".md"

        # D2:markdown 代码块先占位保护,分块后还原,避免代码被拦腰截断
        code_blocks: dict[str, str] = {}
        if is_md:
            text, code_blocks = protect_code_blocks(text)

        # ── 父块切分 ──
        if is_md:
            md_sections = md_splitter.split_text(text)
            raw_parents = []
            for section in md_sections:
                section_text = section.page_content
                # 超过 parent_size 的 section 用通用切分器二次切分，保证父块 ≤ parent_size
                if len(section_text) > parent_size:
                    raw_parents.extend(parent_splitter.split_text(section_text))
                else:
                    raw_parents.append(section_text)
        else:
            raw_parents = parent_splitter.split_text(text)

        if code_blocks:
            raw_parents = [restore_code_blocks(p, code_blocks) for p in raw_parents]

        relative_path = meta.get("relative_path") or Path(meta["file_path"]).name
        file_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]

        seen_parent_texts: set[str] = set()
        for pi, p_text in enumerate(raw_parents):
            if not p_text.strip():
                continue
            # D2:文档内精确重复父块去重(确定性,id 仍按原始序号保持稳定)
            if p_text in seen_parent_texts:
                deduped_parents += 1
                continue
            seen_parent_texts.add(p_text)
            parent_id = f"{file_id}_p{pi}"
            parent_chunks.append({
                "id": parent_id,
                "text": p_text.strip(),
                "source": meta["source"],
                "file_path": meta["file_path"],
            })

            # ── 子块切分（在父块内部再切；代码块整体保留不截断）──
            raw_children = split_with_code_units(p_text, child_splitter)
            seen_child_texts: set[str] = set()
            for ci, c_text in enumerate(raw_children):
                if not c_text.strip():
                    continue
                if c_text in seen_child_texts:
                    deduped_children += 1
                    continue
                seen_child_texts.add(c_text)
                child_chunks.append({
                    "id": f"{parent_id}_c{ci}",
                    "parent_id": parent_id,
                    "text": c_text.strip(),
                    "source": meta["source"],
                    "file_path": meta["file_path"],
                    "chunk_index": ci,
                })

    logger.info(
        "文档分块完成：%d 父块, %d 子块 (parent=%d, child=%d, overlap=%d), "
        "去重:父块 %d / 子块 %d",
        len(parent_chunks), len(child_chunks), parent_size, child_size, overlap,
        deduped_parents, deduped_children,
    )
    return parent_chunks, child_chunks
