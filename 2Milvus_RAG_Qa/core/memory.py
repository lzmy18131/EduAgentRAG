# -*- coding: utf-8 -*-
"""Agent 长期记忆层(借鉴 TencentDB-Agent-Memory 的 4-tier 渐进范式,单用户简化版)。

4-tier 落地:
  ① 事实抽取:对话后 LLM 从用户话中抽求职画像 + 历史事实
  ② 去重/冲突:新非空字段覆盖旧字段(以新为准);事实按 (user, fact) 去重
  ③ 提炼成资产:UserProfile(画像,结构化)+ Fact(历史事实)
  ④ 双写:画像 → MySQL(user_profile);历史事实 → MySQL(user_facts) + Milvus 语义层

召回:查询前读画像 → profile_hint() 生成注入文本;求职类问题由
Skill「求职岗位推荐」按画像条件走 JD 结构化检索;
历史事实按语义检索 search_facts()(Milvus 事实层)。
"""
import hashlib
import json
import math
import re
from datetime import datetime

import pymysql
from pymilvus import DataType

from base.config import cfg
from base.logger import logger

_EXTRACT_PROMPT = (
    "从用户这句话中提取求职画像和历史事实,只输出 JSON(没提到的字段用 null):\n"
    '{{"learning_direction": "学习方向(如java/python/web)", '
    '"target_city": "目标城市", '
    '"expected_salary_k": 期望薪资(纯数字,单位千), '
    '"level": "技术水平(如初级/中级/高级)", '
    '"facts": ["关于用户的事实1", "关于用户的事实2"]}}\n\n'
    "用户的话:{query}"
)

# 事实语义层 Milvus collection(与知识库 edurag_0421 隔离,避免污染检索)
_FACTS_COLLECTION = "edurag_user_facts"
_FACTS_DIM = 1024  # BGE-M3 dense 维度


class MemoryLayer:
    """用户画像记忆:抽取→合并→双写→召回。"""

    def __init__(self, rag_system) -> None:
        self.rag = rag_system
        self._ensure_table()
        self._ensure_facts_table()
        self._ensure_facts_collection()

    # ---------- 存储 ----------
    def _conn(self):
        return pymysql.connect(
            host=cfg.MYSQL_HOST, user=cfg.MYSQL_USER, password=cfg.MYSQL_PASSWORD,
            database=cfg.MYSQL_DATABASE, charset="utf8mb4",
        )

    def _ensure_table(self) -> None:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS user_profile (
            user_id VARCHAR(128) PRIMARY KEY,
            learning_direction VARCHAR(64),
            target_city VARCHAR(64),
            expected_salary_k INT,
            level VARCHAR(32),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
        )
        conn.commit()
        conn.close()

    def get_profile(self, user_id: str) -> dict:
        conn = self._conn()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute(
            "SELECT learning_direction, target_city, expected_salary_k, level "
            "FROM user_profile WHERE user_id=%s",
            (user_id,),
        )
        row = cur.fetchone()
        conn.close()
        return row or {}

    # ---------- 事实层(ME5:双写补全) ----------
    def _ensure_facts_table(self) -> None:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS user_facts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(128) NOT NULL,
            fact VARCHAR(512) NOT NULL,
            fact_type VARCHAR(32) DEFAULT 'chat_fact',
            access_count INT DEFAULT 0,
            last_accessed_at TIMESTAMP NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uk_user_fact (user_id, fact(191))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
        )
        # 优化二:老表补列(幂等,MySQL 8 不支持 ADD COLUMN IF NOT EXISTS,用 try)
        for col_ddl in (
            "ADD COLUMN access_count INT DEFAULT 0",
            "ADD COLUMN last_accessed_at TIMESTAMP NULL",
        ):
            try:
                cur.execute(f"ALTER TABLE user_facts {col_ddl}")
            except Exception:
                pass
        conn.commit()
        conn.close()

    def _ensure_facts_collection(self) -> None:
        """事实语义层 Milvus collection(首次创建 + HNSW 索引)。"""
        client = self.rag._vs._client
        if client.has_collection(_FACTS_COLLECTION):
            client.load_collection(_FACTS_COLLECTION)
            return
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("user_id", DataType.VARCHAR, max_length=128)
        schema.add_field("fact_text", DataType.VARCHAR, max_length=1024)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=_FACTS_DIM)
        schema.add_field("timestamp", DataType.VARCHAR, max_length=32)
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 200},
        )
        client.create_collection(_FACTS_COLLECTION, schema=schema, index_params=index_params)
        client.load_collection(_FACTS_COLLECTION)
        logger.info("事实语义层 collection [%s] 创建完成", _FACTS_COLLECTION)

    @staticmethod
    def _fact_id(user_id: str, fact: str) -> str:
        return hashlib.md5(f"{user_id}::{fact}".encode("utf-8")).hexdigest()[:32]

    def _write_facts(self, user_id: str, facts: list[str]) -> int:
        """双写:MySQL user_facts(去重) + Milvus 语义层(embedding)。"""
        if not facts:
            return 0
        if not user_id:
            logger.warning("_write_facts 拒绝空 user_id(用户隔离)")
            return 0
        conn = self._conn()
        cur = conn.cursor()
        for f in facts:
            cur.execute(
                "INSERT INTO user_facts (user_id, fact) VALUES (%s,%s) "
                "ON DUPLICATE KEY UPDATE fact=VALUES(fact)",
                (user_id, f),
            )
        conn.commit()
        conn.close()

        embs = self.rag._vs._ef(facts)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = [
            {
                "id": self._fact_id(user_id, f),
                "user_id": user_id,
                "fact_text": f,
                "dense_vector": embs["dense"][i].tolist(),
                "timestamp": now,
            }
            for i, f in enumerate(facts)
        ]
        self.rag._vs._client.upsert(collection_name=_FACTS_COLLECTION, data=rows)
        self.rag._vs._client.flush(collection_name=_FACTS_COLLECTION)
        logger.info("事实双写完成: user=%s 共 %d 条", user_id, len(facts))
        return len(facts)

    def get_facts(self, user_id: str) -> list[str]:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("SELECT fact FROM user_facts WHERE user_id=%s", (user_id,))
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows

    @staticmethod
    def _decay_weight(last_accessed_at, now) -> float:
        """优化二:遗忘曲线启发式时序衰减(forgetting-curve-inspired temporal decay)。

        注意(整改后口径):这是自定义的 recency heuristic,不是严格复现
        艾宾浩斯实验模型——7 天后开始衰减,30 天降至 0.63,渐近 0.5(不删除)。
        作为可插拔扩展能力保留,不作为核心默认收益宣称。
        """
        if last_accessed_at is None:
            return 1.0
        days = max(0.0, (now - last_accessed_at).total_seconds() / 86400.0)
        return 0.5 + 0.5 * math.exp(-max(0.0, days - 7.0) / 23.0)

    def search_facts(self, user_id: str, query: str, top_k: int = 3) -> list[str]:
        """语义检索用户历史事实(Milvus 事实层,按 user 过滤)+ 遗忘衰减重排。

        整改(用户隔离):user_id 过滤是**强制**的——查询层 filter
        `user_id == "<escaped>"` 必带;user_id 为空直接返回空,绝不跨用户
        召回(隔离不能只靠 prompt 约定,必须在 schema/查询两层落死)。
        """
        if not user_id:
            logger.warning("search_facts 拒绝无 user_id 的查询(防跨用户泄漏)")
            return []
        try:
            emb = self.rag._vs.encode_query(query)
            escaped = str(user_id).replace("\\", "\\\\").replace('"', '\\"')
            res = self.rag._vs._client.search(
                collection_name=_FACTS_COLLECTION,
                data=[emb.tolist()],
                filter=f'user_id == "{escaped}"',
                anns_field="dense_vector",
                search_params={"metric_type": "IP", "params": {"ef": 64}},
                limit=top_k,
                output_fields=["fact_text"],
            )
            if not res or not res[0]:
                return []
            facts = [h["entity"]["fact_text"] for h in res[0]]
            scores = [float(h["distance"]) for h in res[0]]
        except Exception as e:
            logger.warning("事实语义检索失败: %s", e)
            return []

        # 优化二:召回后读访问统计 → score × 遗忘衰减重排 + 访问计数回写
        now = datetime.now()
        conn = self._conn()
        cur = conn.cursor(pymysql.cursors.DictCursor)
        fmt = ",".join(["%s"] * len(facts))
        cur.execute(
            f"SELECT fact, access_count, last_accessed_at FROM user_facts "
            f"WHERE user_id=%s AND fact IN ({fmt})",
            [user_id] + facts,
        )
        stats = {r["fact"]: r for r in cur.fetchall()}
        ranked = sorted(
            zip(facts, scores),
            key=lambda fs: fs[1] * self._decay_weight(
                (stats.get(fs[0]) or {}).get("last_accessed_at"), now),
            reverse=True,
        )
        ordered = [f for f, _ in ranked]
        try:
            fmt = ",".join(["%s"] * len(ordered))
            cur.execute(
                f"UPDATE user_facts SET access_count=access_count+1, "
                f"last_accessed_at=NOW() WHERE user_id=%s AND fact IN ({fmt})",
                [user_id] + ordered,
            )
            conn.commit()
        except Exception as e:
            logger.warning("事实访问统计回写失败: %s", e)
        conn.close()
        return ordered

    def facts_hint(self, user_id: str, query: str, top_k: int = 3) -> str:
        """按当前问题语义召回历史事实 → 注入 prompt 的文本;无命中返回空串。"""
        facts = self.search_facts(user_id, query, top_k=top_k)
        return ("用户历史事实: " + "; ".join(facts)) if facts else ""

    # ---------- 4-tier 管道 ----------
    def update_from_turn(self, user_id: str, user_query: str) -> dict:
        """① 抽取(LLM)+ ② 冲突解决(新覆盖旧)+ ③④ 双写(MySQL 画像 + MySQL/Milvus 事实)。"""
        prompt = _EXTRACT_PROMPT.format(query=user_query)
        raw = self.rag._call_llm_with_retry(
            prompt, temperature=0, max_tokens=512, reasoning=None, fallback="{}"
        )
        new = self._parse_json(raw)
        old = self.get_profile(user_id)

        merged = {}
        for k in ("learning_direction", "target_city", "level"):
            v = new.get(k) or old.get(k)
            merged[k] = str(v).strip() if v else None
        salary = new.get("expected_salary_k") or old.get("expected_salary_k")
        try:
            salary = int(salary) if salary is not None else None
        except (TypeError, ValueError):
            salary = None

        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_profile "
            "(user_id, learning_direction, target_city, expected_salary_k, level) "
            "VALUES (%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE learning_direction=VALUES(learning_direction), "
            "target_city=VALUES(target_city), "
            "expected_salary_k=VALUES(expected_salary_k), level=VALUES(level)",
            (user_id, merged["learning_direction"], merged["target_city"],
             salary, merged["level"]),
        )
        conn.commit()
        conn.close()

        # 事实层双写(ME5):画像 MySQL 之外,历史事实进 MySQL + Milvus 语义层
        facts = [str(f).strip() for f in (new.get("facts") or []) if str(f).strip()]
        self._write_facts(user_id, facts[:5])

        logger.info("画像更新: %s -> %s", user_id, self.get_profile(user_id))
        return self.get_profile(user_id)

    # ---------- 召回 ----------
    def profile_hint(self, user_id: str) -> str:
        """画像 → 注入 prompt 的文本;无画像返回空串。"""
        p = self.get_profile(user_id)
        parts = [f"{k}:{v}" for k, v in p.items() if v]
        return ("用户画像 " + ", ".join(parts)) if parts else ""

    def recommend_jobs(self, user_id: str, limit: int = 10) -> list[dict]:
        """Skill「求职岗位推荐」:读画像 → JD 结构化检索 → 岗位列表。"""
        from . import jd_structured

        p = self.get_profile(user_id)
        if not p:
            return []
        salary = p.get("expected_salary_k")
        return jd_structured.search_jobs(
            city=p.get("target_city") or None,
            tech=p.get("learning_direction") or None,
            salary_min=int(salary) if salary else None,
            limit=limit,
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        try:
            m = re.search(r"\{.*\}", raw, re.S)
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}
