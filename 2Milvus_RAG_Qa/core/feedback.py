# -*- coding: utf-8 -*-
"""E4 反馈闭环:thumbs 点赞/点踩落库 + 负样本回灌评测集。

链路:
  用户对答案 👍/👎(app.py POST /feedback)
    → 全量写 MySQL 表 user_feedback(可统计)
    → 点踩的 query/answer 追加写 RAG评测/feedback_negatives.jsonl
    → feedback_to_golden.py 把负样本经 LLM 校验后回灌 eval_golden_500.json
      (找不回正确 chunk 的记为知识缺口,写入 feedback_gaps.jsonl)
    → harness 重跑,回归被真实反馈驱动。
"""
import json
import time
from pathlib import Path

import pymysql

from base.config import cfg
from base.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_NEGATIVES_FILE = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "feedback_negatives.jsonl"
_GAPS_FILE = _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "feedback_gaps.jsonl"

_CREATE_SQL = """CREATE TABLE IF NOT EXISTS user_feedback (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(128) NOT NULL DEFAULT '',
  query TEXT NOT NULL,
  answer TEXT,
  rating VARCHAR(8) NOT NULL,
  sources TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY idx_rating (rating),
  KEY idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""


def _conn():
    return pymysql.connect(
        host=cfg.MYSQL_HOST, user=cfg.MYSQL_USER, password=cfg.MYSQL_PASSWORD,
        database=cfg.MYSQL_DATABASE, charset="utf8mb4",
    )


def _ensure_table() -> None:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(_CREATE_SQL)
    conn.commit()
    conn.close()


def record_feedback(
    user_id: str,
    query: str,
    answer: str,
    rating: str,
    sources: list[str] | None = None,
) -> dict:
    """记录一条 👍/👎 反馈:MySQL 全量落库,点踩另写负样本 JSONL。"""
    if rating not in ("up", "down"):
        raise ValueError("rating 必须是 up/down")
    _ensure_table()
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO user_feedback (user_id, query, answer, rating, sources) "
        "VALUES (%s,%s,%s,%s,%s)",
        (user_id or "", query, answer, rating,
         json.dumps(sources or [], ensure_ascii=False)),
    )
    conn.commit()
    fid = cur.lastrowid
    conn.close()

    if rating == "down":
        entry = {
            "query": query,
            "answer": answer,
            "sources": sources or [],
            "user_id": user_id or "",
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(_NEGATIVES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("负反馈已记录: %s", query[:60])
        # 优化四:连续点踩达阈值 → 语义缓存软删除,防坏答案被缓存反复返回
        try:
            from .rag_cache import RagCache
            RagCache().note_downvote(query)
        except Exception as e:
            logger.warning("点踩缓存失效失败: %s", e)
    return {"id": fid, "rating": rating}


def load_negatives() -> list[dict]:
    """读取未处理的负样本 JSONL(去重按 query)。"""
    if not _NEGATIVES_FILE.exists():
        return []
    seen: set[str] = set()
    rows: list[dict] = []
    for line in _NEGATIVES_FILE.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        q = str(item.get("query", "")).strip()
        if q and q not in seen:
            seen.add(q)
            rows.append(item)
    return rows


def remove_processed(queries: set[str]) -> None:
    """从负样本 JSONL 移除已处理条目(回灌脚本幂等重跑)。"""
    if not _NEGATIVES_FILE.exists():
        return
    remaining: list[str] = []
    for line in _NEGATIVES_FILE.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(item.get("query", "")).strip() in queries:
            continue
        remaining.append(line)
    _NEGATIVES_FILE.write_text(
        "\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8"
    )


def feedback_stats() -> dict:
    """反馈统计:总量/好评/差评。"""
    _ensure_table()
    conn = _conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        "SELECT rating, COUNT(*) c FROM user_feedback GROUP BY rating"
    )
    rows = {r["rating"]: r["c"] for r in cur.fetchall()}
    conn.close()
    return {
        "total": sum(rows.values()),
        "up": rows.get("up", 0),
        "down": rows.get("down", 0),
    }


def append_gap(query: str, reason: str) -> None:
    """记录知识缺口(负样本找不到正确 chunk)。"""
    entry = {
        "query": query,
        "reason": reason,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(_GAPS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
