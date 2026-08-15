# -*- coding: utf-8 -*-
"""E4 负样本回灌评测集:点踩反馈 → LLM 校验 → 检索正确 chunk → 追加黄金集。

流程:
  1. 读 feedback_negatives.jsonl(点踩自动写入)
  2. LLM 校验原答案是否真的有错(避免把用户误点当负样本)
  3. 确认为错 → Milvus 检索候选 chunk → LLM 挑出能回答问题的 chunk 并生成标准答案
  4. 找到 → 追加 {question, answer, expected_chunk, source: "user_feedback"} 到黄金集(按 question 去重)
  5. 找不到 → 记入 feedback_gaps.jsonl(知识缺口,供数据治理补语料)
  6. 已处理负样本从 JSONL 移除(幂等,可重复跑)

用法(项目根):
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.RAG评测.feedback_to_golden [--golden <path>] [--limit 20]
"""
import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from base.logger import logger

_feedback = importlib.import_module("2Milvus_RAG_Qa.core.feedback")
_rag_system_mod = importlib.import_module("2Milvus_RAG_Qa.core.rag_system")
_rag_main = importlib.import_module("2Milvus_RAG_Qa.core.rag_main")

_DEFAULT_GOLDEN = str(
    _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden_500.json"
)

_VERIFY_PROMPT = (
    "你是评测数据质检员。判断给定答案对用户问题是否错误或不完整。\n"
    '只输出 JSON:{{"is_wrong": true/false, "reason": "一句话原因"}}\n\n'
    "用户问题:{query}\n\n系统答案:{answer}\n"
)

_PICK_PROMPT = (
    "你是检索评测数据构造专家。从候选知识片段中挑出能正确回答用户问题的片段,"
    "并写出基于该片段的标准答案。若候选片段都回答不了该问题,输出 is_good=false。\n"
    '只输出 JSON:{{"is_good": true/false, "chunk": "选中的片段原文", "answer": "标准答案"}}\n\n'
    "用户问题:{query}\n\n候选片段:\n{candidates}\n"
)


def _parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", (raw or ""), re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def verify_wrong(rag, query: str, answer: str) -> tuple[bool, str]:
    """LLM 校验原答案是否有错。返回 (is_wrong, reason)。"""
    raw = rag._call_llm_with_retry(
        _VERIFY_PROMPT.format(query=query, answer=answer),
        temperature=0, max_tokens=512, reasoning=None, fallback="{}",
    )
    obj = _parse_json(raw)
    return bool(obj.get("is_wrong")), str(obj.get("reason", ""))


def find_correct_chunk(rag, query: str) -> tuple[str, str] | None:
    """检索候选 chunk 并由 LLM 挑出正确片段 + 生成标准答案。"""
    docs = rag._vs.hybrid_search_with_rerank(query, top_k=5)
    if not docs:
        return None
    candidates = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
    raw = rag._call_llm_with_retry(
        _PICK_PROMPT.format(query=query, candidates=candidates),
        temperature=0, max_tokens=2048, reasoning=None, fallback="{}",
    )
    obj = _parse_json(raw)
    if not obj.get("is_good"):
        return None
    chunk = str(obj.get("chunk", "")).strip()
    answer = str(obj.get("answer", "")).strip()
    if not chunk or not answer:
        return None
    return chunk, answer


def replay_negatives(golden_path: str, limit: int = 20) -> dict:
    """把负样本回灌黄金集,返回处理统计。"""
    negatives = _feedback.load_negatives()[:limit]
    if not negatives:
        return {"negatives": 0, "injected": 0, "gaps": 0, "skipped_ok": 0}

    golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    existing = {str(g.get("question", "")).strip() for g in golden}

    vs = _rag_main.init_knowledge_base()
    rag = _rag_system_mod.RAGSystem(vector_store=vs)

    injected = gaps = skipped_ok = 0
    processed_queries: set[str] = set()
    for item in negatives:
        query = str(item.get("query", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not query:
            processed_queries.add("")
            continue
        is_wrong, reason = verify_wrong(rag, query, answer)
        if not is_wrong:
            skipped_ok += 1
            processed_queries.add(query)
            logger.info("负样本校验为误点(答案无错),跳过: %s", query[:50])
            continue
        found = find_correct_chunk(rag, query)
        if found is None:
            _feedback.append_gap(query, reason or "检索不到能回答的 chunk")
            gaps += 1
            processed_queries.add(query)
            logger.info("知识缺口: %s (%s)", query[:50], reason)
            continue
        chunk, std_answer = found
        processed_queries.add(query)
        if query in existing:
            continue  # 已在黄金集(可能先前回灌过),只移除负样本
        golden.append({
            "question": query,
            "answer": std_answer,
            "expected_chunk": chunk,
            "source": "user_feedback",
        })
        existing.add(query)
        injected += 1
        logger.info("回灌黄金集: %s", query[:50])

    Path(golden_path).write_text(
        json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _feedback.remove_processed(processed_queries)
    return {
        "negatives": len(negatives),
        "injected": injected,
        "gaps": gaps,
        "skipped_ok": skipped_ok,
        "golden_total": len(golden),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="负样本回灌黄金集")
    parser.add_argument("--golden", default=_DEFAULT_GOLDEN)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    stats = replay_negatives(args.golden, args.limit)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
