# -*- coding: utf-8 -*-
"""一键 CI 检查:pytest 单元测试 + harness 检索快检(20 条黄金集)。

用法(项目根目录):
    env -u PYTHONPATH <python> ci_check.py

退出码:0 = 全部通过;1 = 存在失败(可直接接入 CI 流水线)。
"""
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_GOLDEN_20 = _ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden_20.json"


def _ensure_golden20() -> None:
    """从 500 条黄金集取前 20 条做 CI 快检(不存在时生成)。"""
    if _GOLDEN_20.exists():
        return
    src = _ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden_500.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    _GOLDEN_20.write_text(
        json.dumps(data[:20], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_step(title: str, cmd: list[str]) -> bool:
    print(f"\n{'=' * 20} {title} {'=' * 20}", flush=True)
    r = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
    tail = (r.stdout or "")[-600:]
    print(tail)
    if r.returncode != 0:
        print(f"[FAIL] {title}: exit={r.returncode}")
        print((r.stderr or "")[-400:])
        return False
    print(f"[PASS] {title}")
    return True


def main() -> None:
    _ensure_golden20()
    py = sys.executable
    ok = True
    ok &= run_step("pytest 单元测试", [py, "-m", "pytest", "-q"])
    ok &= run_step(
        "harness 检索快检(20 条)",
        [py, "-m", "2Milvus_RAG_Qa.RAG评测.eval_harness", "hit-rate",
         "--golden", str(_GOLDEN_20)],
    )
    print(f"\n===== CI 结果: {'全部通过' if ok else '存在失败'} =====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
