# -*- coding: utf-8 -*-
r"""本地小模型调用(快慢分层的"快"层)。

Qwen2.5-0.5B 负责轻活:agent 自省(充分/不充分)、query 改写等短输出任务。
重活(最终生成)仍走云端 deepseek。

模型路径优先 D 盘 ModelCache,回退 C 盘 modelscope 缓存。
"""
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from base.logger import logger

_MODEL_CANDIDATES = [
    r"D:\ModelCache\modelscope\models\Qwen--Qwen2.5-0.5B-Instruct\snapshots\master",
    str(Path.home() / ".cache" / "modelscope" / "models" / "Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / "master"),
]


def _find_model_path() -> str:
    for p in _MODEL_CANDIDATES:
        if Path(p).is_dir():
            return p
    return _MODEL_CANDIDATES[0]


class LocalLLM:
    """本地快模型:自省/改写等轻活(毫秒~百毫秒级)。"""

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        model_path = model_path or _find_model_path()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device
        )
        self._model.eval()
        logger.info("本地快模型加载完成: %s (%s)", model_path, device)

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
        """生成短输出;异常时返回空串(调用方可回退云端)。"""
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            gen = out[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(gen, skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning("本地快模型生成失败: %s", e)
            return ""
