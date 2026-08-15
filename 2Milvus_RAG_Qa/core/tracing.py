# -*- coding: utf-8 -*-
"""轻量 agent 执行轨迹(Tracing):JSONL 落盘,无外部依赖。

替代 Langfuse 自托管:记录 agent 每步事件(query/检索/自省/改写/生成/耗时),
用于黑盒 agent 循环的调试与观测(可被后续 Langfuse 替换,接口不变)。
"""
import json
import time
from pathlib import Path

from base.logger import logger

_DEFAULT_LOG = (
    Path(__file__).resolve().parent.parent.parent / "logs" / "agent_trace.jsonl"
)


class Tracer:
    """JSONL 轨迹记录器。"""

    def __init__(self, log_path=None) -> None:
        self._path = Path(log_path) if log_path else _DEFAULT_LOG
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields) -> None:
        """记录一条事件;异常静默(观测失败不影响主流程)。"""
        try:
            entry = {"ts": round(time.time(), 3), "event": event, **fields}
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("trace 写入失败: %s", e)

    def tail(self, n: int = 20) -> list[dict]:
        """读取最近 n 条事件(调试用)。"""
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines[-n:]]
