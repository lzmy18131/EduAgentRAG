"""检索侧评测 — 生成金标数据集。

从 Milvus 现有 chunk 抽样 N 条（默认 300），每条用 LLM 生成
「问题 + 答案」，连同 chunk 原文作为 expected_chunk，输出 eval_golden.json。

运行：python -m 2Milvus_RAG_Qa.RAG评测.generate_golden_set --num 300
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI
from pymilvus import MilvusClient

from base.config import cfg
from base.logger import logger

_MILVUS_URI = f"http://{cfg.MILVUS_HOST}:{cfg.MILVUS_PORT}"
_DEFAULT_OUTPUT = str(
    _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_golden.json"
)


def _llm_client() -> OpenAI:
    """按 cfg.LLM_* 配置构造 OpenAI 兼容客户端。"""
    return OpenAI(api_key=cfg.LLM_API_KEY, base_url=cfg.LLM_BASE_URL, timeout=120)


def sample_chunks(num: int) -> list[str]:
    """从 Milvus 子块中抽样 num 条文本。

    注：Milvus query 无原生随机抽样，这里取一个 num*5 的候选池后 random.sample，
    兼顾随机性与查询开销（最小合理默认，假设抽样≈候选池随机采样）。
    """
    client = MilvusClient(uri=_MILVUS_URI)
    if cfg.MILVUS_DB_NAME not in client.list_databases():
        raise RuntimeError(f"Milvus 数据库不存在: {cfg.MILVUS_DB_NAME}")
    client.use_database(cfg.MILVUS_DB_NAME)

    # 确保 collection 已加载，已加载时重复调用无副作用
    try:
        client.load_collection(cfg.MILVUS_COLLECTION)
    except Exception:
        pass

    total = client.get_collection_stats(cfg.MILVUS_COLLECTION).get("row_count", 0)
    if total == 0:
        logger.warning("Milvus Collection 为空，无 chunk 可抽样")
        return []

    pool_limit = min(total, num * 5)
    results = client.query(
        collection_name=cfg.MILVUS_COLLECTION,
        filter='chunk_type == "child"',
        output_fields=["text"],
        limit=pool_limit,
    )
    texts = [r.get("text", "") for r in results if r.get("text")]
    return random.sample(texts, min(num, len(texts)))


def _parse_json(content: str) -> tuple[str, str]:
    """从 LLM 返回文本中解析 (question, answer)。"""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM 未返回合法 JSON: {content[:200]}")
    obj = json.loads(text[start:end + 1])
    question = str(obj.get("question", "")).strip()
    answer = str(obj.get("answer", "")).strip()
    if not question or not answer:
        raise ValueError(f"生成的 question/answer 为空: {content[:200]}")
    return question, answer


def generate_one(llm: OpenAI, chunk_text: str) -> tuple[str, str]:
    """基于单个 chunk 生成 (question, answer),失败自动重试 3 次。"""
    prompt = f"""你是一名检索评测数据构造专家。请根据下面的知识片段，生成一个用户可能提出的问题，以及该问题的标准答案。

要求：
1. 问题要自然、口语化，像真实用户会问的
2. 答案要准确、简洁，严格基于知识片段内容
3. 只输出一个 JSON 对象，格式：{{"question": "...", "answer": "..."}}

知识片段：
{chunk_text}"""
    last_err: Exception | None = None
    for attempt in range(1, 4):  # deepseek 偶发空返回,重试 3 次
        try:
            resp = llm.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                reasoning_effort="high",
                max_tokens=1024,
            )
            return _parse_json(resp.choices[0].message.content)
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(2 * attempt)
    raise last_err  # type: ignore[misc]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 RAG 检索评测金标数据集")
    parser.add_argument("--num", type=int, default=300, help="抽样条数（默认 300）")
    parser.add_argument("--output", type=str, default=_DEFAULT_OUTPUT, help="输出 JSON 路径")
    args = parser.parse_args()

    chunks = sample_chunks(args.num)
    logger.info("抽样到 %d 条 chunk，开始 LLM 生成", len(chunks))

    llm = _llm_client()
    golden: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        try:
            question, answer = generate_one(llm, chunk)
            golden.append({
                "question": question,
                "answer": answer,
                "expected_chunk": chunk,
            })
        except Exception as e:
            logger.warning("第 %d 条生成失败，跳过: %s", i, e)
            continue

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"生成 {len(golden)} 条金标，输出到 {output_path}")


if __name__ == "__main__":
    main()
