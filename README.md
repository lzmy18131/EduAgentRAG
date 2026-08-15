# EduAgentRAG — 多模态 Adaptive RAG 智能问答系统

面向 **IT 教育 + 求职辅导** 场景的多模态 Adaptive RAG 问答系统:学员问技术问题走混合向量检索,问"北京 Java 岗 15k+"走 JD 结构化条件检索,贴报错截图走多模态理解——由 LangGraph 循环做「检索→自省→改写→生成」,带意图路由、双路召回、可插拔记忆与评测闭环。

> 检索评测 **HitRate@5 = 0.678(95%CI 0.6333–0.7191)/ MRR = 0.555**(453 条合成回归集,重建语料实测);
> 另有 **56 条人工编写真实口语风格子集**独立报告(分布外探测)。
> 支撑 **358,221 行向量**与 **46,353 条公开招聘 JD**。

## 项目背景

### 目标用户与场景

本项目服务 **IT 培训机构学员** 的两类高频诉求,两者对检索系统的要求截然不同:

| 诉求 | 例子 | 检索本质 | 单路 RAG 的短板 |
|---|---|---|---|
| 技术答疑 | "协程和 goroutine 是什么关系?""这段报错怎么解决?" | **语义匹配**:字面不同但含义相同 | 向量可以;但报错截图无法直接向量化 |
| 求职咨询 | "北京 Java 岗,15k 以上有哪些?" | **精确条件**:城市=北京 AND 薪资≥15k | 向量只保证"语义相似",可能召回"上海 java 8k"——像,但条件错 |

传统单路 RAG 只能二选一:纯向量检索答不准求职条件问题,纯结构化过滤答不了开放技术问题。
本系统用 **意图路由 + 双路召回** 同时服务两类诉求:技术问题走
「BGE-M3 稠密+稀疏 → RRF → bge-reranker-large → MMR」混合检索链;
求职意图(信号词 + 可抽取条件,且排除技术意图)额外触发 JD 结构化检索(MySQL WHERE),
作为**独立槽位(≤3)**前置,不进入 RRF/重排/门控的分数计算——技术类问题零污染;
贴报错截图则由 Qwen-VL 转成结构化 query,复用同一条 RAG 链路。

### 数据来源与规模

- **技术语料**:26 个开源技术仓库(LeetCode 题解 / TensorFlow 官方教程 / Go 文档等,中/英/日/韩/葡 5 语言),清洗 → 父子分块 → BGE-M3 预编码,产物全部落盘可重放;
- **招聘数据**:46,353 条公开招聘 JD,解析为结构化字段(城市/技术方向/薪资区间/经验/学历)入 MySQL;
- **向量库**:Milvus,358,221 行(HNSW 稠密 + 稀疏倒排双索引),全集 584,834 行经确定性抽样 + 黄金父块 100% 保留重建。

### 工程挑战(工程踩坑)

1. **推理模型空返回**:deepseek-v4-pro 的 reasoning 会吃满 max_tokens 导致 content 空——分档 max_tokens + reasoning_effort 必传修复;
2. **异构分数融合**:稠密/稀疏分数量纲差 2 个数量级,加权失效 → RRF 只看排名;
3. **改写回归**:配对消融(McNemar)实测第一版循环显著更差(p=0.039)——改写后重新检索替换原文档,把首轮命中挤出了候选池 → 改写轮改为**并集合并**(只增不减),回归 10→3 条;
4. **门控分数校准**:重排分高度右偏,门控误差大头在分数校准(直过区 FP 29.4%),已量化、校准列入下一步;
5. **评测可信度**:97% LLM 生成的评测集会高估线上效果 → 评测集分层 + 56 条人工编写口语化子集独立报告(实测低 ≈18pp,分布偏移的量化证据)。

## 架构

```
图片/报错截图 ── Qwen-VL ──→ 结构化 query ──┐
文本问题 ──────────────────────────────────┤
                                           ▼
                    LangGraph Adaptive RAG 循环(检索 → 自省 → 改写 → 生成)
                      三级门控(向量路重排分 top1:≥0.7 直过 / ≤0.4 直拒 / 混沌区 0.5B 仲裁)
                      自省/改写:本地 Qwen2.5-0.5B(快,毫秒级)
                      生成:云端 deepseek-v4-pro(强,reasoning=high)
                                           │
              ┌──────────────────┬─────────┴─────────┐
              ▼                  ▼                   ▼
   向量检索(全查询)         JD 结构化检索(仅求职意图)   记忆召回(可插拔)
   BGE-M3 稠密(HNSW)      MySQL 条件过滤         画像注入 + 事实语义召回
   + 稀疏(倒排索引)        (城市/薪资/经验/方向)   (强制 user_id 隔离)
   → RRF(k=60) →          独立槽位 ≤3,置顶标记
   reranker(bge-reranker-large CrossEncoder)
   → MMR(λ=0.7)
              └──────────────────┴─────────┬─────────┘
                                           ▼
   上下文三档组装:≤8 文档且 ≤12k 字符直塞 / 9-15 文档截断 top8 / >15 文档或 >24k 字符分层压缩
                                           ▼
               拒答(向量路 top1<0.15) / 语义缓存 / SSE 流式 / Tracing / 评测闭环
```

> **主链路接线口径**:Web `/chat` 主链路 = Adaptive 循环(自省/改写本地 Qwen2.5-0.5B,
> 任何异常自动降级经典检索;LocalLLM 不可用时回退经典 `RAGSystem.query`);
> `/chat/stream` 走经典流式路径;记忆(画像+历史事实)在两类路径请求前自动注入,
> 记忆层不可用时静默跳过——架构图与运行代码一致。

## 核心特性

| 特性 | 实现 |
|---|---|
| **Adaptive RAG 循环** | LangGraph:检索 → 自省(不充分则改写再检索,≤2 轮)→ 生成;三级门控两端 无额外开销、混沌区 LLM 仲裁;任何异常降级单次检索。说明:消融显示循环不提升检索命中率,价值在意图路由、改写自救与异常兜底 |
| **多路召回** | ① BGE-M3 稠密+稀疏混合检索(HNSW/倒排索引 → RRF → bge-reranker-large → MMR)② JD 结构化条件检索(精确,求职意图定向触发)③ FAQ 语义快筛 |
| **双路并行检索** | 求职意图 无额外开销规则检测(信号词+条件抽取+技术意图排除)→ JD 结构化 + 向量检索线程池并行;消融实测(n=35,带置信区间):条件满足率 32/35→35/35,延迟 695→779ms |
| **多模态** | Qwen-VL 把报错/代码/架构图转成结构化 query,复用同一条 RAG 流水线;md5 缓存同图 无额外开销 |
| **快慢分层** | 本地 Qwen2.5-0.5B 负责自省/改写(毫秒),云端 deepseek-v4-pro 只负责生成 |
| **长期记忆(可插拔扩展模块)** | 借鉴 TencentDB-Agent-Memory 的 4-tier 渐进范式:对话 → 抽取 → 去重 → 双写;画像(MySQL)+ 历史事实(MySQL + Milvus 语义层,强制 user_id 过滤)+ 求职推荐 Skill + LLM-Wiki FAQ |
| **父子分块** | 父块 1200 / 子块 300 / 重叠 80,子块检索、父块回填;代码块整体保留不截断 |
| **数据治理** | 语料审计(精确重复 4.69%/过短 1.18%/语言分布)+ 清洗 + 文档内去重,报告落盘 |
| **可靠性机制** | 置信度拒答(向量路重排分口径)、空返回修复(推理模型 max_tokens×prompt 长度根因)、降级兜底、熔断限流、数据落盘可重放 + 确定性重建、破坏性操作显式化、索引实验可回滚 |
| **评测闭环** | 统一 harness(hit-rate / agent-hit-rate / compare / RAGAS)+ Wilson CI + bootstrap CI + 配对 McNemar 检验 + 指标 diff + 👍/👎 反馈回灌评测集 + CI(`ci_check.py`) |

## 量化指标(真实评测,实测口径)

| 指标 | 数值 | 说明 |
|---|---|---|
| HitRate@5(单次检索,453 条合成回归集) | **0.678** | Wilson 95%CI [0.633, 0.719],重建语料(358,221 行)实测 |
| MRR | **0.555** | bootstrap 95%CI [0.5155, 0.5986](2000 次,seed=42,重采样单位=query) |
| agent 循环 vs 单次(453 条,配对) | **0.6711 vs 0.6777** | 转移矩阵 304/3/0/146,McNemar p=0.25,ΔHitRate 95%CI [-1.3pp, 0.0];改写救活率:合成集 0/45、**口语化子集 1/4=25% 且回归 0**(改写价值在真实分布侧);循环不提升检索命中率,价值=控制层;详见 `agent_compare.json` / `rewrite_rescue_real_style.json` |
| 真实口语风格子集(56 条 human_written) | **独立报告** | 技术 46 条 HitRate@5=0.500(95%CI 0.361-0.639)/MRR 0.406;求职 10 条 JD 条件满足 10/10——口语化问法比合成集低 ≈18pp(分布偏移的量化佐证);见 `eval_real_style_report.json` |
| 向量规模 | **358,221** | 全集 584,834 行 → 按原规模抽样 + 黄金父块 100% 保留(确定性重建) |
| JD 数据 | **46,353 条** | 公开招聘 JD 结构化字段入库(来源/抓取日期/许可见 D 盘语料目录) |
| 语料语言分布 | zh 53.7% / en 31.5% / pt 7.3% / ko 2.6% / ja 0.9% / other 4.0% | 16,000 条抽样审计 |
| 单元测试 | **38 通过** | pytest(31 原有 + 7 新增) |

## 多语言问答

知识库混合了**中 / 英 / 日 / 韩 / 葡**五种语言的技术文档(BGE-M3 支持 100+ 语言),实测同一条检索链路直接服务多语言问题。**口径声明**:各语言仅做了 1 条真实检索的 smoke test(含 2 例跨语言:英问→中文档、韩问→日文档),**不构成系统性多语言评测**;证据落盘 `multilingual_evidence.json`。

## 评测可信度(评测集分层与局限性,声明)

评测集按来源分层、**分指标报告**,不混算、不外推:

| 层 | 构成 | 定位 |
|---|---|---|
| 合成回归集 453 条 | 440 `llm_generated` + 12 `human_reviewed` + 1 `user_feedback`(全带 `source`) | 代码改动的 regression 集。**97% 由 LLM 从真实语料 chunk 生成,表述规范、与语料分布重合,指标会高估线上真实效果**——只用于相对对比,不宣称代表真实用户 |
| dev 308 / blind 145 | seed=42 切分,13 条人工+反馈全进 blind,冻结不参与调参 | 定位为 synthetic held-out,不是完全独立的真实盲测;blind 的 gold 父块在语料重建中 100% 保留,属演进式盲测 |
| 真实口语风格 56 条 | `human_written`:口语化/碎片化/错别字问法(问题仅据标准答案改写,不接触 chunk 文本) | 分布外探测,单独报告 |
| 反馈回灌 | E4 闭环持续增长(当前 1 条) | 机制已建立,样本量待积累——当前不构成统计意义 |

> 已知局限:合成集分布偏移、反向生成泄漏风险、人工复核占比低、反馈样本少;
> 反馈回灌条目(1 条)已进入 blind 子集——用于改系统的反馈样本应重分类为
> regression 集,后续切分会处理(已在文档标注)。
> 后续计划:持续回灌真实用户 query、扩充人工复核比例、混沌区样本已扩至全量。

## 快速开始

```bash
# 1. 依赖
pip install -r requirements.txt

# 2. 配置(config.ini + .env,参照 .env.example)
#    必需:EDU_LLM_API_KEY(deepseek)、EDU_DASHSCOPE_API_KEY(Qwen-VL,多模态)、
#          MySQL / Redis / Milvus 连接

# 3. 启动服务(需 Docker 起 Milvus+Redis:docker compose up -d)
python run.py            # http://127.0.0.1:8000

# 4. 一键评测 / CI / 消融 / 反馈回灌
python ci_check.py
python -m 2Milvus_RAG_Qa.RAG评测.eval_harness hit-rate          # 单次检索
python -m 2Milvus_RAG_Qa.RAG评测.eval_harness agent-hit-rate    # agent 循环
python -m 2Milvus_RAG_Qa.RAG评测.eval_harness compare           # 配对对比(McNemar+转移矩阵)
python -m 2Milvus_RAG_Qa.RAG评测.gate_analysis                  # 门控阈值扫描+敏感性
python -m 2Milvus_RAG_Qa.RAG评测.grade_quality                  # 0.5B 混淆矩阵(全量混沌区)
python -m 2Milvus_RAG_Qa.RAG评测.eval_real_style                # 真实口语风格子集
python -m 2Milvus_RAG_Qa.RAG评测.ablation_2026 s4               # S4 双路消融
python -m 2Milvus_RAG_Qa.RAG评测.ablation_2026 s5               # S5 重排池配对消融
python -m 2Milvus_RAG_Qa.RAG评测.feedback_to_golden             # E4 负样本回灌
python -m 2Milvus_RAG_Qa.core.data_governance                   # D2 语料审计
```

## 消融实验记录

| 实验项 | 对比方式与结果 | 结论 |
|---|---|---|
| 双路并行检索 | 35 条真实求职查询,配对同口径条件判定:仅向量 32/35(91.4%,CI [0.776,0.970])vs 双路 **35/35(100%,CI [0.901,1.000])**;转移矩阵 32/0/3/0,McNemar p=0.25;延迟 695→779ms | 保留:JD 双路零回归、只增不减(+3 条精确条件召回,如"成都 C#"),+84ms 可接受 |
| 重排瘦身 20→15 | 453 条同一批 query 配对跑:pool20=pool15=0.6777,转移矩阵 304/3/3/143,**McNemar p=1.0**,ΔHitRate 95%CI [-1.1pp,+1.1pp](此前双比例 z 与 0.73 系列数字作废) | p=1.0 仅表述"未观察到统计显著差异";收益口径=rerank 候选数减少 25%,非实测 GPU 时间降 25% |
| 向量量化 IVF_PQ | **工程踩坑记录(非消融)**:nlist=1024 构建 >50min 卡 47%;nlist=256 >25min 无进度且卡索引队列(6GB WSL2 受限);两次安全回滚,358,221 行全程完好 | 保留 HNSW;不宣称算法优劣,切换接口保留,大内存机器可重测 |
| 投机解码 | 实测:自省 32token 0.90×、改写 128token 0.67×(更慢),输出无损一致 | 主动弃用:短输出场景草稿+校验开销>收益,红利在长输出场景 |
| JD 隔离围栏 | 围栏机制已落地(`_build_rag_prompt` 显式声明 JD 仅用于求职类问题);原 4 条案例的 0/4 只作机制验证,不宣称统计结论 | 保留为 无额外开销防御(小样本证据已降级表述) |
| 级联仲裁门控 | 0.7 直过/0.4 直拒/混沌区 0.5B 仲裁;阈值扫描 + ±0.05 敏感性 + FP/FN 率(453 条)见 `gate_analysis.json`;混沌区 0.5B 准确率已用全量混沌区样本评测 | 门控分数=向量路重排分,JD 槽位不参与(口径已修正) |

## 使用示例

```python
# Adaptive RAG(核心)
from 2Milvus_RAG_Qa.core.agent_graph import AdaptiveRAG
from 2Milvus_RAG_Qa.core.rag_system import RAGSystem
from 2Milvus_RAG_Qa.core.local_llm import LocalLLM
from 2Milvus_RAG_Qa.core.tracing import Tracer

rag = RAGSystem()                                   # 经典 RAG 内核
agent = AdaptiveRAG(rag, local_llm=LocalLLM(), tracer=Tracer())
result = agent.query("Java 多线程和协程的区别?", verbose=True)
# → {answer, sources, rewrite_count, degraded}

# JD 结构化检索(求职意图定向)
from 2Milvus_RAG_Qa.core.jd_structured import search_jobs
search_jobs(city="北京", tech="java", salary_min=15)  # 北京 Java 15k+ 岗位

# 多模态
from 2Milvus_RAG_Qa.core.multimodal import MultimodalBridge
query = MultimodalBridge().image_to_query(b64, "这个报错怎么解决?")

# 长期记忆(可插拔;user_id 强制隔离)
from 2Milvus_RAG_Qa.core.memory import MemoryLayer
mem = MemoryLayer(rag)
mem.update_from_turn(uid, "我是学Java的,想去北京,期望15k+")   # 抽取画像+事实双写(MySQL+Milvus)
mem.recommend_jobs(uid)                                     # 按画像推荐
mem.search_facts(uid, "帮我推荐北京岗位")                    # 事实层语义召回(user_id 过滤)

# 反馈闭环
from 2Milvus_RAG_Qa.core.feedback import record_feedback, feedback_stats
record_feedback(uid, "北京 java 岗位", answer, "down")       # 点踩→负样本 JSONL
feedback_stats()                                             # 反馈统计
```

## 项目结构

```
├── app.py                    # FastAPI 接入层(/chat、/chat/stream SSE、/upload、/feedback)
├── run.py / MAIN.py          # 启动入口 / CLI
├── ci_check.py               # 一键 CI(pytest + 检索快检)
├── rebuild_corpus.py         # 确定性重建(显式 --rebuild,数据可重放)
├── 1MySQL_qa/                # FAQ 语义快筛子系统
├── 2Milvus_RAG_Qa/
│   ├── core/
│   │   ├── agent_graph.py    # LangGraph Adaptive RAG 循环(检索→自省→改写→生成)
│   │   ├── vector_store.py   # 混合检索(稠密HNSW+稀疏→RRF→reranker→MMR)+ 上下文三档组装 + 索引切换
│   │   ├── jd_structured.py  # JD 结构化检索
│   │   ├── dual_retrieval.py # S4 双路并行检索(求职意图路由+独立槽位)
│   │   ├── multimodal.py     # Qwen-VL 多模态桥接
│   │   ├── memory.py         # 4-tier 长期记忆(画像+事实双写+user_id 隔离)
│   │   ├── llm_wiki.py       # ME4 LLM-Wiki 知识提炼
│   │   ├── feedback.py       # E4 反馈闭环(MySQL + 负样本)
│   │   ├── data_governance.py# D2 语料审计/清洗/代码块保护
│   │   ├── local_llm.py      # 本地快模型(快慢分层)
│   │   ├── tracing.py        # 执行轨迹(JSONL 可观测)
│   │   ├── rag_system.py     # 经典 RAG 主流程(意图→策略→检索→生成)
│   │   └── strategy_selector.py  # 检索策略选择
│   └── RAG评测/              # harness(McNemar/CI) / gate_analysis / grade_quality /
│                             # 真实口语风格子集 / 消融 / 反馈回灌 / 黄金集(带来源)
├── base/                     # 配置与日志
├── static/config.ini         # 参数配置
└── tests/                    # pytest 38 项
```

## 技术栈

**检索**:Milvus、BGE-M3(稠密+稀疏)、bge-reranker-large(CrossEncoder)、MySQL、Redis
**编排**:LangGraph、FastAPI(SSE 流式)
**模型**:deepseek-v4-pro(云端生成,reasoning_effort 必传)、Qwen-VL(多模态)、Qwen2.5-0.5B(本地轻活)
**评测**:pytest、hit_rate/MRR、RAGAS、自研 harness(Wilson/bootstrap/McNemar)+ CI

## License

MIT
