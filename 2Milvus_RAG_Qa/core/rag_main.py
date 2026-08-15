"""RAG 系统命令行入口 — 知识库初始化 + 交互问答。

启动方式:
    python -m 2Milvus_RAG_Qa.core.rag_main
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from base.logger import logger
from .document_loader import process_documents
from .rag_system import RAGSystem
from .vector_store import VectorStore


def init_knowledge_base(rebuild: bool = False) -> VectorStore:
    """初始化知识库：加载文档 → 分块 → 写入 Milvus。

    Args:
        rebuild: 是否清空重建

    Returns:
        初始化好的 VectorStore 实例
    """
    vs = VectorStore()

    if rebuild:
        vs.clear()

    if vs.count_chunks() > 0:
        logger.info("知识库已有 %d 条记录，跳过导入", vs.count_chunks())
        return vs

    logger.info("知识库为空，开始导入文档...")
    parent_chunks, child_chunks = process_documents()
    if not child_chunks:
        logger.warning("未找到可导入的文档，请检查 data/ 目录")
        return vs

    vs.add_documents(parent_chunks, child_chunks)
    return vs


def main() -> None:
    """命令行交互入口。"""
    print("=" * 60)
    print("  EdeRAG 智慧问答系统 — RAG 检索增强模块")
    print("=" * 60)

    try:
        vs = init_knowledge_base()
        rag = RAGSystem(vector_store=vs)
    except Exception as e:
        logger.error("系统初始化失败: %s", e)
        print(f"❌ 初始化失败: {e}")
        return

    print(f"\n📚 知识库: {vs.count_chunks()} 条记录")
    print("📖 输入问题 | exit/退出 终止")
    print("-" * 60)

    while True:
        try:
            q = input("\n🔍 请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if q.lower() in ("exit", "quit", "退出", "q"):
            print("👋 再见！")
            break
        if not q:
            continue

        result = rag.query(q, verbose=True)
        print(f"\n📋 [{result['intent']}] [{result['strategy']}] [{result['latency_ms']}ms]")
        print(f"✅ {result['answer']}")
        if result["sources"]:
            print(f"📎 参考来源: {len(result['sources'])} 条文档")


if __name__ == "__main__":
    main()
