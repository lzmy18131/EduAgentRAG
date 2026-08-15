"""OCR 文档加载器 — 基于 PaddleOCR (PP-OCRv5) 的扫描件/图片文字提取。

专为教育文档优化：
    - 图片 (png/jpg) 直接 OCR
    - 扫描型 PDF（无文字层）→ 逐页渲染 → OCR
    - 多语言混合排版（中/英/日/韩）识别

PaddleOCR 模型首次调用自动下载（PP-OCRv5 det/rec/cls），
初始化较慢，因此采用懒加载 + 模块级单例。
"""

import os
import sys
from pathlib import Path

# ── sys.path 注入，确保项目根可访问 ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from base.logger import logger

# 支持 OCR 的图片扩展名
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class PaddleOCRExtractor:
    """PaddleOCR 文本提取器 — 模块级单例，懒加载。"""

    _instance: "PaddleOCRExtractor | None" = None

    def __new__(cls) -> "PaddleOCRExtractor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ocr = None
        return cls._instance

    @property
    def ocr(self):
        """懒加载 PaddleOCR 引擎（首次调用才初始化）。

        paddleocr 3.x API：使用 PP-OCRv5 系列模型（det/rec/orientation）。
        """
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                text_recognition_batch_size=8,
                # paddle 3.x 在 Windows CPU 有 PIR 静态图 bug，用 onnxruntime 后端绕开
                engine="onnxruntime",
            )
            logger.info("PaddleOCR (PP-OCRv5) 引擎加载完成")
        return self._ocr

    @staticmethod
    def _parse_results(results) -> list[str]:
        """解析 paddlex 3.x predict 返回结果，提取识别文本行。"""
        lines: list[str] = []
        for res in results or []:
            # paddlex Result 对象：res.rec_texts / res.rec_scores
            texts = getattr(res, "rec_texts", None)
            if texts is None and isinstance(res, dict):
                texts = res.get("rec_texts")
            for t in texts or []:
                if t and str(t).strip():
                    lines.append(str(t).strip())
        return lines

    def extract_from_image(self, image_path: str | Path) -> str:
        """单张图片 OCR，返回拼接后的文本。"""
        results = self.ocr.predict(str(image_path))
        return "\n".join(self._parse_results(results))

    def extract_from_pdf(self, pdf_path: str | Path) -> str:
        """扫描型 PDF：逐页渲染为图片后 OCR。

        Returns:
            全部页面的识别文本；PDF 无页面时返回空串。
        """
        import pymupdf

        pdf_path = str(pdf_path)
        doc = pymupdf.open(pdf_path)
        all_text: list[str] = []
        for page_num, page in enumerate(doc):
            # 渲染为 200 DPI 图片（平衡速度与识别率）
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            results = self.ocr.predict(img_bytes)
            all_text.extend(self._parse_results(results))
            logger.info("PDF OCR 第 %d/%d 页完成", page_num + 1, doc.page_count)
        doc.close()
        return "\n".join(all_text)

    def extract(self, file_path: str | Path) -> str:
        """按扩展名自动选择提取方式。"""
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self.extract_from_pdf(file_path)
        if suffix in _IMAGE_EXTS:
            return self.extract_from_image(file_path)
        raise ValueError(f"不支持的 OCR 文件类型: {suffix}")


# 模块级单例 — 全局唯一，懒加载
ocr_extractor = PaddleOCRExtractor()


def ocr_document(file_path: str | Path) -> str:
    """对单个文件执行 OCR 提取文本。"""
    return ocr_extractor.extract(file_path)
