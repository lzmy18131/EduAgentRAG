"""日志模块 — 提供控制台+文件双输出日志器。

使用方式:
    from base.logger import logger
    logger.info("这是一条日志")
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


def get_logger(
    log_name: str = "app",
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB 单文件上限
    backup_count: int = 5,              # 保留最近 5 个备份
) -> logging.Logger:
    """创建并返回日志器（控制台 + 按天滚动的文件）。

    特性：
        - 控制台实时输出
        - 文件按天命名，自动归档
        - 单文件超过 max_bytes 自动轮转
        - 重复调用返回同一个 logger 实例（单例模式）
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(log_name)
    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    # 日志格式：时间 - 模块 - 级别 - 文件:行号 - 消息
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ========== 控制台输出 ==========
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ========== 文件输出（按天 + 按大小轮转）==========
    log_filename = os.path.join(
        log_dir, f"{log_name}_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = RotatingFileHandler(
        filename=log_filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# 全局日志实例 — 模块导入即用
logger = get_logger()
