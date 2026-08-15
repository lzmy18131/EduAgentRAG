"""全局配置模块 — 读取 config.ini 统一管理所有参数。

敏感信息（API Key）支持环境变量覆盖，优先于 config.ini。

使用方法:
    from base.config import cfg
    print(cfg.MYSQL_HOST)
"""

import os
from configparser import ConfigParser
from pathlib import Path

# 加载项目根目录的 .env 文件（敏感配置）
from dotenv import load_dotenv
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


class Config:
    """应用配置单例，从 config.ini 加载所有参数。

    配置项按 section 分组：
        - mysql: 数据库连接参数
        - redis: 缓存连接参数
        - milvus: 向量库连接参数
        - llm: 大模型 API 参数
        - retrieval: 检索分块参数
        - logger: 日志路径
        - app: 业务配置

    API Key 加载优先级：环境变量 EDU_LLM_API_KEY > config.ini
    """

    def _get(self, section: str, key: str) -> str:
        """读取配置，``EDU_<SECTION>_<KEY>`` 环境变量优先。"""
        env_name = f"EDU_{section.upper()}_{key.upper()}"
        env_value = os.getenv(env_name)
        return env_value if env_value is not None else self._parser.get(section, key)

    def _getint(self, section: str, key: str) -> int:
        """读取整数配置，并支持环境变量覆盖。"""
        return int(self._get(section, key))

    def _getfloat(self, section: str, key: str) -> float:
        """读取浮点配置，并支持环境变量覆盖。"""
        return float(self._get(section, key))

    def __init__(self, config_path: str | None = None):
        project_root = Path(__file__).resolve().parent.parent
        if config_path is None:
            config_path = str(project_root / "static" / "config.ini")

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        self._parser = ConfigParser()
        self._parser.read(config_path, encoding="utf-8")

        # ==================== MySQL 配置 ====================
        self.MYSQL_HOST = self._get("mysql", "host")
        self.MYSQL_USER = self._get("mysql", "user")
        self.MYSQL_PASSWORD = self._get("mysql", "password")
        self.MYSQL_DATABASE = self._get("mysql", "database")

        # ==================== Redis 配置 ====================
        self.REDIS_HOST = self._get("redis", "host")
        self.REDIS_PORT = self._getint("redis", "port")
        self.REDIS_PASSWORD = self._get("redis", "password") or None
        self.REDIS_DB = self._getint("redis", "db")

        # ==================== Milvus 向量库配置 ====================
        self.MILVUS_HOST = self._get("milvus", "host")
        self.MILVUS_PORT = self._getint("milvus", "port")
        self.MILVUS_DB_NAME = self._get("milvus", "database_name")
        self.MILVUS_COLLECTION = self._get("milvus", "collection_name")

        # ==================== LLM 大模型配置 ====================
        self.LLM_MODEL = self._get("llm", "model")
        self.LLM_API_KEY = self._get("llm", "api_key") or os.getenv("LLM_API_KEY") or ""
        if not self.LLM_API_KEY:
            self.LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.LLM_BASE_URL = self._get("llm", "base_url")

        # RAGAS 评测的 embedding 服务与 DeepSeek 对话服务分开配置。
        self.EVAL_EMBEDDING_MODEL = os.getenv("EDU_EVAL_EMBEDDING_MODEL", "")
        self.EVAL_EMBEDDING_API_KEY = os.getenv("EDU_EVAL_EMBEDDING_API_KEY", "")
        self.EVAL_EMBEDDING_BASE_URL = os.getenv("EDU_EVAL_EMBEDDING_BASE_URL", "")

        # ==================== 检索分片参数 ====================
        self.PARENT_CHUNK_SIZE = self._parser.getint("retrieval", "parent_chunk_size")
        self.CHILD_CHUNK_SIZE = self._parser.getint("retrieval", "child_chunk_size")
        self.CHUNK_OVERLAP = self._parser.getint("retrieval", "chunk_overlap")
        self.RETRIEVAL_K = self._parser.getint("retrieval", "retrieval_k")
        self.CANDIDATE_M = self._parser.getint("retrieval", "candidate_m")
        self.DENSE_WEIGHT = self._getfloat("retrieval", "dense_weight")
        self.SPARSE_WEIGHT = self._getfloat("retrieval", "sparse_weight")
        self.RERANK_POOL_MULTIPLIER = self._getint("retrieval", "rerank_pool_multiplier")
        self.RANKER_TYPE = self._get("retrieval", "ranker_type").strip().lower()
        self.RRF_K = self._getint("retrieval", "rrf_k")

        # 上下文压缩开关（D10：句子级相关度过滤，压到 ~2000 字符）
        self.CONTEXT_COMPRESS_ENABLED = self._parser.getboolean(
            "retrieval", "context_compress_enabled", fallback=True
        )
        self.CONTEXT_MAX_CHARS = self._parser.getint(
            "retrieval", "context_max_chars", fallback=2000
        )

        # ==================== Web 管理接口 ====================
        self.UPLOAD_API_KEY = os.getenv("EDU_APP_UPLOAD_API_KEY", "")

        # ==================== 日志配置 ====================
        log_file = self._parser.get("logger", "log_file")
        if not os.path.isabs(log_file):
            log_file = str(project_root / log_file)
        self.LOG_FILE = log_file


# 全局配置实例 — 模块导入时即初始化
cfg = Config()
