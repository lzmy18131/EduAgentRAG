# -*- coding: utf-8 -*-
"""ME4 LLM-Wiki:技术语料提炼 FAQ 式结构化知识(借鉴 TencentDB-Agent-Memory)。

与原文检索的分工:
  知识库 chunk 是"原文",长、杂、冗余;LLM-Wiki 把同一主题的多个 chunk
  提炼成 {question, answer, keywords} 的 FAQ 资产——不是原文压缩,而是
  跨 chunk 归纳的结构化知识,供快速问答与知识缺口发现。

用法(项目根):
  # 从 Milvus 抽样 chunk 提炼(真实语料)
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.core.llm_wiki build \
      --topic "Go 位运算" --sample-queries "golang 位运算 and not" --num-chunks 6
  # 检索 FAQ
  env -u PYTHONPATH <python> -m 2Milvus_RAG_Qa.core.llm_wiki search --q "golang &^"
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pymysql

from base.config import cfg
from base.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CREATE_SQL = """CREATE TABLE IF NOT EXISTS llm_wiki (
  id INT AUTO_INCREMENT PRIMARY KEY,
  question VARCHAR(512) NOT NULL,
  answer TEXT NOT NULL,
  keywords VARCHAR(512),
  topic VARCHAR(64),
  source_chunks TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_question (question(191))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""

_DISTILL_PROMPT = (
    "你是技术知识提炼专家。把下面的技术文档片段提炼成 FAQ 式结构化知识条目。\n"
    "要求:\n"
    "1. 每条 FAQ 由 question(用户会问的自然问题)、answer(基于片段的准确简洁答案,"
    "50字内)、keywords(检索关键词,逗号分隔3-5个) 三个字段组成\n"
    "2. 归纳跨片段的共性,不要逐段复述原文\n"
    "3. 只输出 JSON 数组,每个元素格式为 {{\"question\": ..., \"answer\": ..., \"keywords\": ...}},2-4 条\n\n"
    "主题:{topic}\n\n文档片段:\n{chunks}\n"
)


def _conn():
    return pymysql.connect(
        host=cfg.MYSQL_HOST, user=cfg.MYSQL_USER, password=cfg.MYSQL_PASSWORD,
        database=cfg.MYSQL_DATABASE, charset="utf8mb4",
    )


class LLMWiki:
    """技术语料 → FAQ 式结构化知识资产。"""

    def __init__(self, rag_system=None) -> None:
        self.rag = rag_system  # 供 LLM 调用;None 时仅检索可用
        self._ensure_table()

    def _ensure_table(self) -> None:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(_CREATE_SQL)
        conn.commit()
        conn.close()

    # ---------- 提炼 ----------
    def distill(self, topic: str, chunks: list[str]) -> list[dict]:
        """LLM 把同主题 chunk 提炼为 FAQ 条目并落库(去重按 question)。"""
        if not chunks:
            return []
        raw = self.rag._call_llm_with_retry(
            _DISTILL_PROMPT.format(
                topic=topic,
                chunks="\n\n".join(f"[{i + 1}] {c[:600]}" for i, c in enumerate(chunks)),
            ),
            temperature=0.1, max_tokens=2048, reasoning=None, fallback="[]",
        )
        entries = self._parse_entries(raw)
        if not entries:
            logger.warning("LLM-Wiki 提炼为空: %s", raw[:100])
            return []

        conn = _conn()
        cur = conn.cursor()
        saved = 0
        for e in entries:
            q = str(e.get("question", "")).strip()
            a = str(e.get("answer", "")).strip()
            if not q or not a:
                continue
            cur.execute(
                "INSERT INTO llm_wiki (question, answer, keywords, topic, source_chunks) "
                "VALUES (%s,%s,%s,%s,%s) "
                "ON DUPLICATE KEY UPDATE answer=VALUES(answer), "
                "keywords=VALUES(keywords), topic=VALUES(topic)",
                (q, a, str(e.get("keywords", ""))[:512], topic,
                 json.dumps(chunks, ensure_ascii=False)[:4000]),
            )
            saved += 1
        conn.commit()
        conn.close()
        logger.info("LLM-Wiki 提炼落库: 主题=%s 新增/更新 %d 条", topic, saved)
        return entries

    @staticmethod
    def _parse_entries(raw: str) -> list[dict]:
        try:
            m = re.search(r"\[.*\]", raw or "", re.S)
            if not m:
                return []
            data = json.loads(m.group(0))
            return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []
        except Exception:
            return []

    # ---------- 检索 ----------
    def search(self, question: str, limit: int = 5) -> list[dict]:
        """多词评分检索 FAQ:每命中一个词 +1 分,按分数排序返回。"""
        conn = _conn()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        terms = [t for t in re.split(r"[\s,，。?？!！]+", question) if len(t) >= 2]
        if not terms:
            conn.close()
            return []
        score_exprs, params = [], []
        for t in terms[:4]:
            score_exprs.append("(keywords LIKE %s OR question LIKE %s OR answer LIKE %s)")
            params.extend([f"%{t}%", f"%{t}%", f"%{t}%"])
        score_sql = " + ".join(score_exprs)
        cur.execute(
            f"SELECT question, answer, keywords, topic, ({score_sql}) AS score "
            f"FROM llm_wiki HAVING score > 0 ORDER BY score DESC LIMIT %s",
            params + [limit],
        )
        rows = cur.fetchall()
        conn.close()
        return rows

    def count(self) -> int:
        conn = _conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM llm_wiki")
        total = cur.fetchone()[0]
        conn.close()
        return total


def _sample_chunks_from_milvus(queries: list[str], per_query: int) -> list[str]:
    """按主题 query 从知识库检索 chunk(真实语料抽样)。"""
    rag_main = __import__("2Milvus_RAG_Qa.core.rag_main", fromlist=["init_knowledge_base"])
    vs = rag_main.init_knowledge_base()
    chunks: list[str] = []
    for q in queries:
        for text in vs.hybrid_search_with_rerank(q, top_k=per_query):
            if text not in chunks:
                chunks.append(text)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-Wiki 知识提炼")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--topic", required=True)
    p_build.add_argument("--sample-queries", default="", help="逗号分隔的检索 query 用于抽样 chunk")
    p_build.add_argument("--num-chunks", type=int, default=6)
    p_build.add_argument("--chunks-file", default="", help="或直接从 JSON 文件读 chunks")
    p_search = sub.add_parser("search")
    p_search.add_argument("--q", required=True)
    args = parser.parse_args()

    if args.cmd == "build":
        rag_system = __import__("2Milvus_RAG_Qa.core.rag_system", fromlist=["RAGSystem"])
        wiki = LLMWiki(rag_system.RAGSystem())
        if args.chunks_file:
            chunks = json.loads(Path(args.chunks_file).read_text(encoding="utf-8"))
        else:
            queries = [q.strip() for q in args.sample_queries.split(",") if q.strip()]
            if not queries:
                queries = [args.topic]
            chunks = _sample_chunks_from_milvus(queries, max(1, args.num_chunks // len(queries)))
        print(f"抽样 chunk 数: {len(chunks)}")
        entries = wiki.distill(args.topic, chunks)
        print(json.dumps({"topic": args.topic, "entries": entries}, ensure_ascii=False, indent=2))
    else:
        wiki = LLMWiki()
        rows = wiki.search(args.q)
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
