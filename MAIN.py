"""EdeRAG 总 CLI：MySQL 快筛、Milvus RAG、DeepSeek 生成。"""

import argparse
import importlib
from pathlib import Path

from base.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parent


def init_vector_store(rebuild: bool = False):
    """复用 RAG 模块的知识库初始化流程。"""
    rag_main = importlib.import_module("2Milvus_RAG_Qa.core.rag_main")
    return rag_main.init_knowledge_base(rebuild=rebuild)


class EduRAGSystem:
    """MySQL BM25 快筛后降级到 Milvus RAG。"""

    def __init__(self) -> None:
        mysql_mod = importlib.import_module("1MySQL_qa.mysql_qa_main")
        rag_mod = importlib.import_module("2Milvus_RAG_Qa.core.rag_system")
        self._mysql = mysql_mod.MySQLQaSystem()
        self._vs = init_vector_store()
        self._rag = rag_mod.RAGSystem(vector_store=self._vs)
        logger.info("EduRAGSystem 初始化完成")

    def query(self, text: str) -> dict:
        answer, msg = self._mysql.search(text)
        if answer:
            return {"answer": answer, "source": "mysql_bm25", "msg": msg}
        result = self._rag.query(text)
        source = f"rag_{result['strategy']}" if result["sources"] else "llm_direct"
        return {
            "answer": result["answer"],
            "source": source,
            "msg": f"检索{len(result['sources'])}条文档, {result['latency_ms']}ms",
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EdeRAG 智慧问答系统")
    sub = parser.add_subparsers(dest="command", required=True)
    query_cmd = sub.add_parser("query", help="执行完整问答")
    query_cmd.add_argument("text", nargs="+", help="用户问题")
    sub.add_parser("rebuild", help="清空并重建 Milvus 知识库")
    sub.add_parser("stats", help="查看 Milvus 知识库记录数")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "rebuild":
        vs = init_vector_store(rebuild=True)
        print(f"重建完成: {vs.count_chunks()} 条记录")
        return
    if args.command == "stats":
        vs = init_vector_store()
        print(f"知识库记录数: {vs.count_chunks()}")
        return

    result = EduRAGSystem().query(" ".join(args.text))
    print(f"\n[{result['source']}] {result['msg']}")
    print(result["answer"])


if __name__ == "__main__":
    main()
