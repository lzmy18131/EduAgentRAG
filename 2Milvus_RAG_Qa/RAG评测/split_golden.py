# -*- coding: utf-8 -*-
"""优化六:评测集盲测隔离——dev(调参)/ blind(隔离) 拆分。

规则(固定种子可复现):
  · human_reviewed(12)+ user_feedback(1) 全部进 blind 集——
    它们最接近"非 LLM 自产"的独立标签;
  · 其余 llm_generated 440 条按 70/30 随机拆 dev/blind。
输出:eval_dev.json / eval_blind.json + 拆分统计。

边界(说明):
  · 严格意义上的盲测(真实线上用户日志改写)需要日志积累;当前 blind 主要
    成分仍为 LLM 生成条目,但与 13 条人工/反馈条目一起在本轮调参中隔离;
  · E4 反馈闭环正在持续把真实用户问题回灌进 goldset(source=user_feedback),
    blind 集的真实成分会随线上运行增长——这是"演进式盲测"。
用法:env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.split_golden
"""
import json
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_GOLDEN = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden_500.json"
_DEV = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_dev.json"
_BLIND = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_blind.json"
_SEED = 42
_DEV_RATIO = 0.7


def main() -> None:
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    independent = [x for x in golden if x.get("source") in ("human_reviewed", "user_feedback")]
    llm_gen = [x for x in golden if x.get("source") != "human_reviewed" and x.get("source") != "user_feedback"]

    rng = random.Random(_SEED)
    llm_gen = list(llm_gen)
    rng.shuffle(llm_gen)
    cut = int(len(llm_gen) * _DEV_RATIO)
    dev = llm_gen[:cut]
    blind = independent + llm_gen[cut:]

    _DEV.write_text(json.dumps(dev, ensure_ascii=False, indent=2), encoding="utf-8")
    _BLIND.write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "total": len(golden),
        "dev": len(dev),
        "blind": len(blind),
        "blind_独立来源(人工复核+用户反馈)": len(independent),
        "seed": _SEED,
        "说明": "blind 主要成分为 LLM 生成条目;真实用户日志条目随 E4 反馈闭环持续回灌增长",
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
