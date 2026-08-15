# -*- coding: utf-8 -*-
"""D3 黄金集来源标注:为评测集标注真实来源,保证评测可信。

来源分类:
  llm_generated    LLM 从真实语料 chunk 生成的问题+标准答案(generate_golden_set.py)
  user_feedback    反馈闭环回灌的真实用户问题(feedback_to_golden.py)
  human_reviewed   上述来源中经人工抽查复核的条目

用法(项目根):
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.annotate_golden_sources \
      --golden 2Milvus_RAG_Qa/RAG评测/eval_golden_500.json \
      --human-file <question 列表文件,每行一个问题,精确匹配>
"""
import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def annotate(golden_path: str, human_questions: list[str] | None = None) -> dict:
    """给黄金集逐条补 source 字段(幂等,已有 source 且非空则保留)。"""
    path = Path(golden_path)
    golden = json.loads(path.read_text(encoding="utf-8"))
    human = {q.strip() for q in (human_questions or []) if q.strip()}
    human_matched: set[str] = set()
    stats: dict[str, int] = {}

    for item in golden:
        q = str(item.get("question", "")).strip()
        if item.get("source"):
            stats[item["source"]] = stats.get(item["source"], 0) + 1
            continue
        source = "llm_generated"
        if q in human:
            source = "human_reviewed"
            human_matched.add(q)
        item["source"] = source
        stats[source] = stats.get(source, 0) + 1

    path.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    unmatched = human - human_matched
    return {"total": len(golden), "distribution": stats, "unmatched_human": sorted(unmatched)}


def main() -> None:
    parser = argparse.ArgumentParser(description="黄金集来源标注")
    parser.add_argument("--golden", required=True)
    parser.add_argument("--human-file", default="", help="人工复核问题列表(每行一个)")
    args = parser.parse_args()

    human = []
    if args.human_file:
        human = Path(args.human_file).read_text(encoding="utf-8").splitlines()
    result = annotate(args.golden, human)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
