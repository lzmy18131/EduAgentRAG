# -*- coding: utf-8 -*-
"""多模态桥接:图片 → Qwen-VL → 结构化检索 query(复用现有 RAG 流水线)。

设计决策:图片统一交给 VLM 一步到位(不搞 OCR/文档解析的精细路由),
输出"适合 RAG 检索的技术问题描述",直接作为后续检索的 query。

依赖:.env 中的 EDU_DASHSCOPE_API_KEY(DashScope OpenAI 兼容端点)。
"""
import base64
import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from base.logger import logger

_VL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-vl-plus"

_PROMPT = (
    "用户发来一张图片(可能是报错截图、代码截图、架构图、表格或文档截图)。\n"
    "请把图片内容转换为一段适合知识库检索的技术问题描述,包含关键信息"
    "(报错类型、异常信息、技术名词、涉及的技术栈等)。\n"
    "只输出问题描述,不要任何解释或前缀。\n\n"
    "用户附带的问题:{user_text}"
)


class MultimodalBridge:
    """图片 → 结构化 query 的 VLM 桥接器(图片 hash 缓存由调用方负责)。"""

    def __init__(self, model: str | None = None) -> None:
        load_dotenv()  # 确保 .env 已加载
        key = os.getenv("EDU_DASHSCOPE_API_KEY", "")
        if not key:
            raise RuntimeError("EDU_DASHSCOPE_API_KEY 未配置,多模态功能不可用")
        self._client = OpenAI(api_key=key, base_url=_VL_BASE_URL, timeout=60)
        self._model = model or os.getenv("EDU_VL_MODEL", _DEFAULT_MODEL)
        self._cache: dict[str, str] = {}  # M2:图片 hash 缓存,同图只调一次 VLM
        logger.info("多模态桥接就绪: model=%s", self._model)

    def image_to_query(self, image_base64: str, user_text: str | None = None) -> str:
        """图片(+可选用户文字)→ 适合 RAG 检索的技术问题描述。

        M2 图片 hash 缓存:同一张图(相同内容)只调一次 VLM,直接命中缓存。
        """
        img_hash = hashlib.md5(image_base64.encode("ascii", "ignore")).hexdigest()
        if img_hash in self._cache:
            logger.info("多模态图片缓存命中: %s", img_hash[:8])
            return self._cache[img_hash]
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            },
            {"type": "text", "text": _PROMPT.format(user_text=user_text or "无")},
        ]
        resp = self._client.chat.completions.create(
            model=self._model, messages=[{"role": "user", "content": content}], max_tokens=512
        )
        result = (resp.choices[0].message.content or "").strip()
        self._cache[img_hash] = result
        return result

    @staticmethod
    def file_to_base64(path: str) -> str:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
