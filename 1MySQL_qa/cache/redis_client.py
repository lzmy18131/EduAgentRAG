"""Redis 缓存客户端 — 提供 JSON 自动序列化的读写接口。

使用方式:
    from cache.redis_client import redis_client
    redis_client.set_data("key", {"data": 123})
    value = redis_client.get_data("key")
"""

import json
from typing import Any

import redis
from base.config import cfg
from base.logger import logger


class RedisClient:
    """Redis 客户端封装，自动处理 JSON 序列化/反序列化。

    特性：
        - set_data() 自动将 Python 对象序列化为 JSON 字符串存储
        - get_data() 自动反序列化回 Python 对象
        - get_answer() 按 query 快速读取问答缓存
    """

    def __init__(self) -> None:
        try:
            self._conn = redis.Redis(
                host=cfg.REDIS_HOST,
                port=cfg.REDIS_PORT,
                password=cfg.REDIS_PASSWORD,
                db=cfg.REDIS_DB,
                decode_responses=True,  # 自动解码为 str，无需手动 bytes→str
                socket_connect_timeout=5,
            )
            # 验证连接
            self._conn.ping()
            logger.info("Redis 连接初始化成功")
        except Exception as e:
            logger.error(f"Redis 连接失败: {e}，缓存功能将不可用")
            self._conn = None

    @property
    def available(self) -> bool:
        """Redis 是否可用。"""
        return self._conn is not None

    def set_data(self, key: str, value: Any, expire: int | None = None) -> bool:
        """写入缓存，自动将 value 序列化为 JSON。

        Args:
            key: 缓存键
            value: 任意可 JSON 序列化的 Python 对象
            expire: 过期时间（秒），None 表示永久

        Returns:
            写入成功返回 True
        """
        if not self._conn:
            logger.warning("Redis 不可用，写入跳过")
            return False
        try:
            json_val = json.dumps(value, ensure_ascii=False)
            if expire:
                self._conn.setex(key, expire, json_val)
            else:
                self._conn.set(key, json_val)
            return True
        except Exception as e:
            logger.error(f"Redis 写入失败 key={key}: {e}")
            return False

    def get_data(self, key: str) -> Any | None:
        """读取缓存，自动反序列化 JSON。

        Args:
            key: 缓存键

        Returns:
            反序列化后的 Python 对象，不存在返回 None
        """
        if not self._conn:
            return None
        try:
            raw = self._conn.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Redis 读取失败 key={key}: {e}")
            return None

    def delete_data(self, *keys: str) -> int:
        """删除缓存键，Redis 不可用时返回 0。"""
        if not self._conn or not keys:
            return 0
        try:
            return int(self._conn.delete(*keys))
        except Exception as e:
            logger.error(f"Redis 删除失败: {e}")
            return 0

    def get_answer(self, query: str) -> str | None:
        """根据用户问题查询缓存的答案。

        Args:
            query: 用户输入的问题文本

        Returns:
            缓存的答案字符串，未命中返回 None
        """
        cache_key = f"rag_answer:{query}"
        return self.get_data(cache_key)

    # ────────────────── FAQ 版本号缓存 ──────────────────

    def get_faq_version(self) -> int:
        """读取 FAQ 语料版本号，缺省返回 1。"""
        if not self._conn:
            return 1
        try:
            raw = self._conn.get("faq:version")
            return int(raw) if raw is not None else 1
        except Exception as e:
            logger.error(f"读取 faq:version 失败: {e}")
            return 1

    def bump_faq_version(self) -> int:
        """自增 FAQ 语料版本号（INCR），返回新版本号。"""
        if not self._conn:
            return 1
        try:
            return int(self._conn.incr("faq:version"))
        except Exception as e:
            logger.error(f"自增 faq:version 失败: {e}")
            return 1

    @staticmethod
    def _faq_answer_key(norm_key: str, version: int) -> str:
        """构造 FAQ 答案缓存键: faq:v{version}:ans:{norm_key}。"""
        return f"faq:v{version}:ans:{norm_key}"

    def get_faq_answer(self, norm_key: str, version: int) -> str | None:
        """读取指定版本的 FAQ 答案缓存，未命中返回 None。"""
        if not self._conn:
            return None
        key = self._faq_answer_key(norm_key, version)
        try:
            return self._conn.get(key)
        except Exception as e:
            logger.error(f"Redis 读取失败 key={key}: {e}")
            return None

    def set_faq_answer(
        self, norm_key: str, version: int, answer: str, ttl: int = 86400
    ) -> bool:
        """写入 FAQ 答案缓存，带 TTL，成功返回 True。"""
        if not self._conn:
            logger.warning("Redis 不可用，写入跳过")
            return False
        key = self._faq_answer_key(norm_key, version)
        try:
            return bool(self._conn.setex(key, ttl, answer))
        except Exception as e:
            logger.error(f"Redis 写入失败 key={key}: {e}")
            return False


# 全局 Redis 客户端单例
redis_client = RedisClient()
