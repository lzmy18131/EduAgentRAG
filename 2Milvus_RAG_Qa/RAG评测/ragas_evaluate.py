"""RAGAS 评测脚本 — 自动化评估 RAG 系统的检索和生成质量。

四大指标：
    - Context Precision（上下文精度）：检索的文档和问题相关性
    - Context Recall（上下文召回）：标准答案信息是否被检索覆盖
    - Faithfulness（忠实度）：回答是否来自检索文档，有无幻觉
    - Answer Relevancy（答案相关性）：回答是否贴合问题

运行：python -m 2Milvus_RAG_Qa.RAG评测.ragas_evaluate
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
# DeepSeek 兼容：API 仅支持 n=1，将 answer_relevancy 的候选生成数限制为 1
answer_relevancy.strictness = 1
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from base.config import cfg
from base.logger import logger


def run_evaluation(eval_data_path: str | None = None) -> dict:
    """执行 RAGAS 评测。

    预期数据格式（JSON 或 CSV）：
        question     — 用户问题
        answer       — 系统生成的回答
        contexts     — 检索到的上下文列表（JSON 字符串或 \n 分隔）
        ground_truth — 标准答案

    Returns:
        各指标得分字典
    """
    # 初始化 LLM（用于判断忠实度和答案相关性）
    # DeepSeek 推理模型响应慢，超时放宽到 180s，避免 ragas 并发调用超时
    llm = LangchainLLMWrapper(ChatOpenAI(
        model=cfg.LLM_MODEL,
        api_key=cfg.LLM_API_KEY,
        base_url=cfg.LLM_BASE_URL,
        temperature=0,
        request_timeout=180,
        # DeepSeek 推理模型（deepseek-v4-pro）不传 reasoning_effort 会 100% 空返回，
        # 必须显式传 reasoning_effort="low"，否则 RAGAS 的 LLM 打分（faithfulness/answer_relevancy）
        # 会全部落空。
        model_kwargs={"reasoning_effort": "low"},
    ))

    # DeepSeek 对话 API 不提供 OpenAI embedding，优先使用独立 embedding 服务；
    # 未配置时回退到本地 BGE-M3（modelscope 缓存），保证离线可评测。
    if all((
        cfg.EVAL_EMBEDDING_MODEL,
        cfg.EVAL_EMBEDDING_API_KEY,
        cfg.EVAL_EMBEDDING_BASE_URL,
    )):
        logger.info("使用远程 embedding 服务: %s", cfg.EVAL_EMBEDDING_MODEL)
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            model=cfg.EVAL_EMBEDDING_MODEL,
            api_key=cfg.EVAL_EMBEDDING_API_KEY,
            base_url=cfg.EVAL_EMBEDDING_BASE_URL,
        ))
    else:
        _local = Path.home() / ".cache" / "modelscope" / "models" / "BAAI--bge-m3" / "snapshots" / "master"
        _model_path = str(_local) if _local.is_dir() else "BAAI/bge-m3"
        logger.info("未配置远程 embedding，回退本地 BGE-M3: %s", _model_path)
        emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
            model_name=_model_path,
            model_kwargs={"device": "cpu"},
        ))

    # 加载评测数据
    eval_path = eval_data_path or str(
        _PROJECT_ROOT / "2Milvus_RAG_Qa" / "RAG评测" / "eval_data_ragas.json"
    )
    suffix = Path(eval_path).suffix.lower()
    if suffix == ".json":
        dataset = Dataset.from_json(eval_path)
    elif suffix == ".csv":
        dataset = Dataset.from_csv(eval_path)
    else:
        raise ValueError("评测数据仅支持 .json 或 .csv")

    # 将 contexts 字符串转为列表
    def _parse_contexts(example: dict) -> dict:
        ctx = example.get("contexts", "")
        if isinstance(ctx, str):
            example["contexts"] = [c.strip() for c in ctx.split("\n") if c.strip()]
        return example

    dataset = dataset.map(_parse_contexts)

    # 执行评估
    # DeepSeek 推理模型慢且并发易触发 Connection error：降并发到 2、放宽单样本超时
    logger.info("开始 RAGAS 评测，共 %d 条测试数据", len(dataset))
    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(timeout=600, max_workers=2),
    )

    import numpy as np

    def _safe_mean(vals):
        """跳过 None/NaN（连接失败样本）后取均值；全部失败返回 NaN。"""
        cleaned = [
            v for v in vals
            if v is not None and not (isinstance(v, float) and v != v)
        ]
        if not cleaned:
            return float("nan")
        return float(np.mean(cleaned))

    # ragas 返回每个样本的分数列表，取均值聚合（鲁棒跳过失败项）
    scores = {
        "context_precision": _safe_mean(result["context_precision"]),
        "context_recall": _safe_mean(result["context_recall"]),
        "faithfulness": _safe_mean(result["faithfulness"]),
        "answer_relevancy": _safe_mean(result["answer_relevancy"]),
    }

    logger.info("评测完成: %s", scores)
    return scores


def main() -> None:
    """评测入口，输出格式化结果。

    支持命令行传入数据路径与报告路径：
        python -m 2Milvus_RAG_Qa.RAG评测.ragas_evaluate [数据json] [报告json]
    """
    eval_path = sys.argv[1] if len(sys.argv) > 1 else None
    report_path = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        scores = run_evaluation(eval_path)
        print("\n" + "=" * 50)
        print("  RAGAS 评测结果")
        print("=" * 50)
        for metric, score in scores.items():
            if score != score:  # NaN 检测
                print(f"  {metric:<20s}  {'N/A (LLM 超时或无结果)':<30s}")
                continue
            bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
            print(f"  {metric:<20s}  {bar} {score:.3f}")
        print("=" * 50)

        if report_path:
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            Path(report_path).write_text(
                json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"报告已写入 {report_path}")
    except Exception as e:
        logger.exception("RAGAS 评测失败")
        raise SystemExit(f"评测失败: {e}") from e


if __name__ == "__main__":
    main()
