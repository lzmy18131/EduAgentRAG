# -*- coding: utf-8 -*-
"""S4 双路并行检索:求职意图 → JD 结构化(MySQL) + 向量检索(Milvus)并行,合并文档。

动机:
  - 求职类查询("北京 java 15k 岗位")是"精确条件"问题,向量检索只会找语义相似,
    JD 结构化 WHERE 过滤才准;但两类信息都可能有用(岗位 + 相关技术/就业知识)。
  - 串行会先等向量检索再查 JD,浪费 JD 检索的毫秒级延迟;
    两条路相互独立,用 ThreadPoolExecutor 并行发,总延迟 ≈ max(两路) 而非 sum。

规则抽取(无额外开销):
  条件抽取走规则(城市词表 / 技术方向别名 / 薪资正则),不额外调 LLM——
  若为这条并行路径再串一次云端 LLM 抽取,就抵消了并行的收益。

合并策略(整改后):
  JD 命中是精确条件匹配,作为**独立结构化槽位**(≤JD_MAX_SLOTS)置于
  向量重排结果之前;score=1.0 仅是结构化置顶标记,不参与自省门控——
  门控分数只取 source=="vector" 的重排分(异构分数不混排)。
  JD 路仅在 should_trigger_jd(信号词+可抽取条件)时触发,技术类问题零污染。
"""
import re
from concurrent.futures import ThreadPoolExecutor

from base.logger import logger

# 常驻 2 线程池:避免每轮检索新建 executor 的开销(实测每轮新建 ≈ 并行收益被吃平)
_POOL = ThreadPoolExecutor(max_workers=2)

# JD 结构化结果在上下文中的独立槽位上限(整改:明确置顶规则——
# JD 不进入 RRF/重排池,作为独立结构化源,最多占前 JD_MAX_SLOTS 个槽位,
# 且仅在求职意图下触发;技术类问题零污染)
JD_MAX_SLOTS = 3

# 求职意图信号词(无额外开销规则检测)
_JOB_SIGNALS = (
    "岗位", "招聘", "求职", "找工作", "工作机会", "职位", "offer", "jd",
    "面试", "就业", "薪资", "工资", "待遇", "五险", "社保", "双休",
    "简历", "跳槽", "换工作", "应届", "上班",
)

# 城市词表(覆盖 job_data 中出现频率最高的城市)
_CITY_WORDS = (
    "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "西安",
    "苏州", "长沙", "郑州", "重庆", "天津", "青岛", "大连", "厦门", "福州",
    "合肥", "济南", "昆明", "南昌", "贵阳", "南宁", "石家庄", "哈尔滨",
    "沈阳", "东莞", "佛山", "无锡", "宁波", "温州", "珠海", "海口", "兰州",
    "太原", "乌鲁木齐", "三亚", "常州", "徐州", "烟台", "潍坊", "洛阳",
    "唐山", "泉州", "惠州",
)

# 技术方向别名 → job_data.tech_direction 的 5 个枚举值(实测全表仅 5 类)
_TECH_ALIASES = (
    (("java", "javase", "javaee"), "java"),
    (("python",), "python"),
    (("前端", "web", "网页", "h5", "vue", "react", "javascript", "js"), "web"),
    (("linux", "运维"), "linux"),
    (("c#", "csharp", ".net"), "C#"),
)


def is_job_query(query: str) -> bool:
    """无额外开销求职意图检测:含任一信号词即视为求职类查询。"""
    q = query.lower()
    return any(s in q for s in _JOB_SIGNALS)


# 技术意图强信号(整改:区分"面试时问什么技术原理"这类技术问题)
# 当查询只有技术方向条件、无城市/薪资条件且带这些标记时,判为技术问题,
# 不触发 JD 路,避免无关岗位污染技术类召回。
_TECH_QUERY_MARKERS = (
    "原理", "代码", "报错", "怎么用", "是什么意思", "教程", "文档", "语法",
    "源码", "debug", "实现", "区别", "什么是",
)


def should_trigger_jd(query: str, intent: str | None = None) -> bool:
    """JD 结构化检索的触发条件(整改:意图路由收紧)。

    触发 = (求职信号词 OR 已分类的求职意图) AND 可抽取至少一个结构化条件
    AND 排除"技术意图强信号"——例如"面试时会问哪些 java 原理"含信号词
    "面试",但带技术标记"原理"且无城市/薪资条件,判为技术问题,不触发 JD。
    """
    job_intent = is_job_query(query) or intent in {"就业薪资", "求职岗位推荐"}
    if not job_intent:
        return False
    cond = extract_jd_conditions(query)
    if not cond:
        return False
    has_geo_salary = bool(cond.get("city") or cond.get("salary_min") or cond.get("salary_max"))
    if not has_geo_salary and any(m in query for m in _TECH_QUERY_MARKERS):
        return False
    return True


def extract_jd_conditions(query: str) -> dict:
    """从查询中规则抽取 JD 结构化条件(城市/技术方向/薪资区间)。

    例: "北京 java 15k-20k 的岗位" → {"city": "北京", "tech": "java",
    "salary_min": 15, "salary_max": 20}
    """
    cond: dict = {}
    for c in _CITY_WORDS:
        if c in query:
            cond["city"] = c
            break
    ql = query.lower()
    for aliases, tech in _TECH_ALIASES:
        for a in aliases:
            if a in ql:
                cond["tech"] = tech
                break
        if "tech" in cond:
            break
    m = re.search(r"(\d+)\s*k\s*[-~到至]\s*(\d+)\s*k", query)
    if m:
        cond["salary_min"], cond["salary_max"] = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"(\d+)\s*k", query)
        if m:
            cond["salary_min"] = int(m.group(1))
    return cond


def format_jd_text(row: dict) -> str:
    """JD 行 → 注入上下文的紧凑文本。"""
    return (
        f"【招聘岗位】{row.get('title', '')} | 城市 {row.get('city', '')} | "
        f"公司 {row.get('company', '')} | 技术方向 {row.get('tech_direction', '')} | "
        f"薪资 {row.get('salary', '')} | 经验 {row.get('experience', '')} | "
        f"学历 {row.get('education', '')}"
    )


def merge_docs(prev: list[dict], new: list[dict], top_k: int) -> list[dict]:
    """多轮检索候选合并(整改:改写只增不减,消除改写回归)。

    配对消融(agent_compare.json 第一版)发现改写路径净损失 8 个命中
    (regression 10 vs rescue 2,McNemar p=0.039)——原因是改写后重新检索
    **替换**了原文档,把首轮已命中的父块挤出了候选池。
    修复:按文本去重取并集,同文本保留最高分;JD 结构化槽位保持在最前
    (≤JD_MAX_SLOTS,不参与分数排序),向量文档按分数降序取 top_k——
    改写只增不减,保证最终候选 ⊇ 首轮候选(与子问题检索的并集策略一致)。

    Args:
        prev:  上一轮合并后的 docs(首轮为空列表)
        new:   本轮新检索 docs
        top_k: 向量文档保留数量

    Returns:
        合并后的 docs:JD 槽位在前,向量文档按分数降序 ≤ top_k。
    """
    jd_docs = [d for d in list(prev) + list(new) if d.get("source") == "jd"]
    vec_map: dict[str, dict] = {}
    for d in list(prev) + list(new):
        if d.get("source") == "jd":
            continue
        text = d.get("text", "")
        if text not in vec_map or d.get("score", 0.0) > vec_map[text].get("score", 0.0):
            vec_map[text] = dict(d)
    seen: set[str] = set()
    jd_out: list[dict] = []
    for d in jd_docs:
        if d["text"] not in seen:
            seen.add(d["text"])
            jd_out.append(d)
        if len(jd_out) >= JD_MAX_SLOTS:
            break
    vec_out = sorted(vec_map.values(), key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]
    return jd_out + vec_out


def dual_retrieve(
    query: str,
    vector_search,
    top_k: int,
    jd_limit: int | None = None,
    intent: str | None = None,
) -> list[dict]:
    """求职意图下并行走 JD 结构化 + 向量检索,合并为统一 doc 列表。

    Args:
        query:         用户问题
        vector_search: callable(query, top_k) -> list[(text, score)]
        top_k:         向量检索 top_k
        jd_limit:      JD 返回上限,默认 JD_MAX_SLOTS(独立槽位,不挤占向量名额)
        intent:        已分类的意图(rag_system 传入);"就业薪资"等可补足
                      信号词漏检的求职意图

    Returns:
        [{"text": str, "score": float, "source": "jd"|"vector"}, ...]
        JD 命中在前(score 记 1.0 仅为结构化置顶标记,**不参与自省门控**——
        门控只取 source=="vector" 的重排分),向量文档在后。

    合并策略(整改后明确口径):
        JD 结构化结果是**独立的精确条件源**,在向量检索完成重排后按固定
        槽位(≤JD_MAX_SLOTS)前置;它不进入 RRF 融合,也不进入 reranker/MMR
        池——与向量结果分数不可比,因此不做统一排序。
    """
    from . import jd_structured

    jd_limit = jd_limit if jd_limit is not None else JD_MAX_SLOTS
    vec_docs: list = []
    jd_rows: list = []

    if should_trigger_jd(query, intent=intent):
        cond = extract_jd_conditions(query)
        logger.info("双路并行检索触发: cond=%s", cond)
        f_vec = _POOL.submit(vector_search, query, top_k)
        f_jd = _POOL.submit(jd_structured.search_jobs, **cond, limit=jd_limit)
        vec_docs = f_vec.result() or []
        jd_rows = f_jd.result() or []
    else:
        vec_docs = vector_search(query, top_k) or []

    merged: list[dict] = []
    for row in jd_rows:
        merged.append({"text": format_jd_text(row), "score": 1.0, "source": "jd"})
    for text, score in vec_docs:
        merged.append({"text": text, "score": float(score), "source": "vector"})
    logger.info(
        "双路检索完成: JD=%d + 向量=%d → 合并 %d",
        len(jd_rows), len(vec_docs), len(merged),
    )
    return merged
