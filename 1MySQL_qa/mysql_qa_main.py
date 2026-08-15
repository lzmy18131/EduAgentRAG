"""MySQL 高频问答模块 CLI。

命令：
    python -m 1MySQL_qa.mysql_qa_main init-db
    python -m 1MySQL_qa.mysql_qa_main query "问题"
    python -m 1MySQL_qa.mysql_qa_main interactive
"""

import argparse
import importlib
import time
from pathlib import Path

_DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "JP学科知识问答.csv"


def _mysql_client():
    return importlib.import_module("1MySQL_qa.db.mysql_client").mysql_client


def _redis_client():
    return importlib.import_module("1MySQL_qa.cache.redis_client").redis_client


class MySQLQaSystem:
    """基于语义向量 + MySQL 的高频问答系统。"""

    SEARCH_THRESHOLD = 0.90

    def __init__(self) -> None:
        search_cls = importlib.import_module(
            "1MySQL_qa.retrieval.faq_semantic"
        ).FaqSemanticSearch
        self._searcher = search_cls(_redis_client(), _mysql_client())

    def search(self, query: str) -> tuple[str | None, str]:
        start = time.perf_counter()
        answer, msg = self._searcher.search(query, threshold=self.SEARCH_THRESHOLD)
        elapsed = time.perf_counter() - start
        return answer, f"{msg}，耗时 {elapsed:.4f}s"


def init_database(csv_path: str) -> int:
    """创建数据库、表并幂等导入 CSV，成功后自增 FAQ 语料版本号。"""
    count = _mysql_client().initialize(csv_path)
    _redis_client().bump_faq_version()
    return count


def interactive() -> None:
    qa_system = MySQLQaSystem()
    print("输入问题进行检索 | 输入 exit / 退出 终止程序")
    while True:
        try:
            query = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit", "退出", "q"}:
            break
        if not query:
            continue
        answer, msg = qa_system.search(query)
        print(f"状态: {msg}")
        print(f"答案: {answer}" if answer else "未命中，将降级到 RAG")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EdeRAG MySQL 高频问答管理")
    sub = parser.add_subparsers(dest="command", required=True)
    init_cmd = sub.add_parser("init-db", help="创建数据库和表，并导入问答 CSV")
    init_cmd.add_argument("--csv", default=str(_DEFAULT_CSV), help="问答 CSV 路径")
    query_cmd = sub.add_parser("query", help="执行一次语义问答")
    query_cmd.add_argument("text", help="用户问题")
    sub.add_parser("interactive", help="进入交互问答")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = None
    try:
        if args.command == "init-db":
            count = init_database(args.csv)
            print(f"初始化完成，导入或更新 {count} 条问答")
        elif args.command == "query":
            answer, msg = MySQLQaSystem().search(args.text)
            print(f"状态: {msg}")
            print(f"答案: {answer}" if answer else "未命中，将降级到 RAG")
        else:
            interactive()
        client = _mysql_client()
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
