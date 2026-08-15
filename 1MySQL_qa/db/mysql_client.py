"""MySQL 数据库客户端，支持延迟连接和首次初始化。"""

from pathlib import Path

import pandas as pd
import pymysql
from pymysql.cursors import DictCursor

from base.config import cfg
from base.logger import logger


class MySqlClient:
    """管理 ``subject_qa`` 问答表，导入数据时按问题幂等更新。"""

    def __init__(self) -> None:
        self._conn: pymysql.Connection | None = None
        self._cursor: pymysql.cursors.Cursor | None = None

    def _connect(self, include_database: bool = True) -> pymysql.Connection:
        kwargs = {
            "host": cfg.MYSQL_HOST,
            "user": cfg.MYSQL_USER,
            "password": cfg.MYSQL_PASSWORD,
            "charset": "utf8mb4",
            "connect_timeout": 10,
            "cursorclass": DictCursor,
        }
        if include_database:
            kwargs["database"] = cfg.MYSQL_DATABASE
        return pymysql.connect(**kwargs)

    def connect(self) -> None:
        """按需连接业务数据库。"""
        if self.connected:
            return
        self.close()
        self._conn = self._connect(include_database=True)
        self._cursor = self._conn.cursor()
        logger.info("MySQL 数据库连接成功")

    def _ensure_connection(self) -> None:
        if not self.connected:
            self.connect()

    @property
    def connected(self) -> bool:
        return self._conn is not None and bool(self._conn.open)

    def create_database(self) -> None:
        """连接 MySQL 服务并创建配置中的数据库。"""
        database = cfg.MYSQL_DATABASE
        if not database.replace("_", "").isalnum():
            raise ValueError(f"非法数据库名: {database}")
        conn = self._connect(include_database=False)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            conn.commit()
            logger.info("MySQL 数据库 [%s] 就绪", database)
        finally:
            conn.close()

    def create_table(self) -> None:
        """创建问答表；``question`` 使用唯一索引保证重复导入安全。"""
        self._ensure_connection()
        sql = """
            CREATE TABLE IF NOT EXISTS subject_qa (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                question    VARCHAR(1000) NOT NULL,
                answer      TEXT NOT NULL,
                create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_subject_qa_question (question)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        try:
            self._cursor.execute(sql)
            self._conn.commit()
            logger.info("数据表 subject_qa 就绪")
        except Exception:
            self._conn.rollback()
            logger.exception("建表失败")
            raise

    def initialize(self, csv_path: str | None = None) -> int:
        """创建数据库和表，可选导入 CSV，返回处理行数。"""
        self.create_database()
        self.connect()
        self.create_table()
        return self.insert_from_csv(csv_path) if csv_path else 0

    def insert_from_csv(self, csv_path: str) -> int:
        """导入含 ``question``、``answer`` 列的 UTF-8 CSV。"""
        self._ensure_connection()
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV 文件不存在: {path}")
        df = pd.read_csv(path, encoding="utf-8")
        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("CSV 缺少必需的 'question' 或 'answer' 列")

        rows = [
            (str(row["question"]).strip(), str(row["answer"]).strip())
            for _, row in df.iterrows()
            if pd.notna(row["question"]) and pd.notna(row["answer"])
            and str(row["question"]).strip() and str(row["answer"]).strip()
        ]
        sql = """
            INSERT INTO subject_qa (question, answer) VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE answer = VALUES(answer)
        """
        try:
            self._cursor.executemany(sql, rows)
            self._conn.commit()
            logger.info("从 CSV 导入或更新 %d 条记录: %s", len(rows), path)
            return len(rows)
        except Exception:
            self._conn.rollback()
            logger.exception("CSV 导入失败")
            raise

    def fetch_questions(self) -> list[str]:
        self._ensure_connection()
        self._cursor.execute("SELECT question FROM subject_qa ORDER BY id")
        rows = self._cursor.fetchall()
        return [row["question"] for row in rows]

    def fetch_answer(self, question: str) -> str | None:
        self._ensure_connection()
        self._cursor.execute(
            "SELECT answer FROM subject_qa WHERE question = %s LIMIT 1", (question,)
        )
        row = self._cursor.fetchone()
        return row["answer"] if row else None

    def close(self) -> None:
        if self._cursor:
            self._cursor.close()
        if self._conn:
            self._conn.close()
        self._cursor = None
        self._conn = None


mysql_client = MySqlClient()
