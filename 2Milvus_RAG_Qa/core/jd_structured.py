# -*- coding: utf-8 -*-
"""JD 结构化检索:解析 IT 岗位招聘 markdown → MySQL 结构化字段 → 精确条件查询。

与向量检索的分工:
  - 向量检索:语义/知识类问题(技术问答)
  - JD 结构化:求职类"精确条件"问题(城市=北京 AND 薪资>=15k AND 方向=java)

字段解析规则(实测 46353 条零缺失):
  薪资 "5k-10k" → salary_min=5, salary_max=10(单位千)
  经验 "3-5年本科" → experience="3-5年", education="本科"
  技术方向 "C%23" → url 解码为 "C#"
"""
import re
from urllib.parse import unquote

import pymysql

from base.config import cfg
from base.logger import logger

_TABLE = "job_data"
_CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS `{_TABLE}` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  city VARCHAR(64),
  company VARCHAR(255),
  tech_direction VARCHAR(64),
  salary_min INT,
  salary_max INT,
  experience VARCHAR(64),
  education VARCHAR(32),
  message TEXT,
  KEY idx_city (city),
  KEY idx_tech (tech_direction),
  KEY idx_salary (salary_min)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def _field(block: str, name: str) -> str:
    m = re.search(rf"- {name}:([^\n]*)", block)
    return m.group(1).strip() if m else ""


def _parse_salary(s: str) -> tuple[int | None, int | None]:
    """'5k-10k' -> (5, 10);其它格式返回 (None, None)。"""
    m = re.match(r"(\d+)\s*k\s*-\s*(\d+)\s*k", s, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _parse_experience(e: str) -> tuple[str, str]:
    """'3-5年本科' -> ('3-5年', '本科');'不限本科' -> ('不限', '本科')。"""
    m = re.match(r"(\d+\s*-\s*\d+年)(.+)", e)
    if m:
        return m.group(1).replace(" ", ""), m.group(2)
    m2 = re.match(r"(不限)(.+)", e)
    if m2:
        return "不限", m2.group(2)
    return e, ""


def _conn():
    return pymysql.connect(
        host=cfg.MYSQL_HOST, user=cfg.MYSQL_USER, password=cfg.MYSQL_PASSWORD,
        database=cfg.MYSQL_DATABASE, charset="utf8mb4",
    )


def build_and_load(md_path: str, batch: int = 1000) -> int:
    """解析 JD markdown 并全量入库(先建表)。返回入库条数。"""
    txt = open(md_path, encoding="utf-8").read()
    blocks = txt.split("## ")[1:]  # 首元素是文件标题,丢弃

    conn = _conn()
    cur = conn.cursor()
    cur.execute(_CREATE_SQL)
    cur.execute(f"DELETE FROM `{_TABLE}`")  # 幂等:重跑前清空
    conn.commit()

    rows = []
    for b in blocks:
        title = b.split("\n", 1)[0].strip()
        city = _field(b, "城市")
        company = _field(b, "公司")
        tech = unquote(_field(b, "技术方向"))
        salary = _field(b, "薪资")
        exp_raw = _field(b, "经验要求")
        salary_min, salary_max = _parse_salary(salary)
        experience, education = _parse_experience(exp_raw)
        rows.append((
            title, city, company, tech, salary_min, salary_max,
            experience, education, b.strip(),
        ))

    sql = (
        f"INSERT INTO `{_TABLE}` (title, city, company, tech_direction, "
        f"salary_min, salary_max, experience, education, message) "
        f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    )
    for i in range(0, len(rows), batch):
        cur.executemany(sql, rows[i:i + batch])
        conn.commit()
    cur.execute(f"SELECT COUNT(*) FROM `{_TABLE}`")
    total = cur.fetchone()[0]
    conn.close()
    logger.info("JD 结构化入库完成: %d 条 -> 表 %s", total, _TABLE)
    return total


def search_jobs(
    city: str | None = None,
    tech: str | None = None,
    salary_min: int | None = None,
    salary_max: int | None = None,
    experience: str | None = None,
    education: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """结构化精确查询,支持城市/技术方向/薪资区间/经验/学历过滤。"""
    conds = []
    params = []
    if city:
        conds.append("city=%s")
        params.append(city)
    if tech:
        conds.append("tech_direction=%s")
        params.append(tech)
    if salary_min is not None:
        conds.append("salary_max>=%s")  # 岗位薪资上限 >= 用户期望下限,即"≥15k"
        params.append(salary_min)
    if salary_max is not None:
        conds.append("salary_min<=%s")
        params.append(salary_max)
    if experience and experience != "不限":
        conds.append("experience=%s")
        params.append(experience)
    if education:
        conds.append("education=%s")
        params.append(education)
    where = (" WHERE " + " AND ".join(conds)) if conds else ""

    conn = _conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        f"SELECT title, city, company, tech_direction, salary_min, salary_max, "
        f"experience, education FROM `{_TABLE}`{where} LIMIT %s",
        params + [limit],
    )
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        r["salary"] = f"{r['salary_min']}k-{r['salary_max']}k"
    return rows
