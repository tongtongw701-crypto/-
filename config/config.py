#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集中配置 — 从 .env 读取，设置默认值，创建必要目录
"""

import os
from pathlib import Path

# 自动加载 .env
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


# LLM 厂商预设（OpenAI 兼容接口，仅 base_url 不同）
LLM_PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "dashscope": {
        "name": "阿里云 DashScope (通义千问)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-max",
    },
    "moonshot": {
        "name": "月之暗面 (Kimi)",
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
    },
    "zhipu": {
        "name": "智谱 AI (GLM)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4",
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
}


class Config:
    """配置类 — 从环境变量读取，提供默认值"""

    def __init__(self):
        root = Path(__file__).parent.parent.absolute()
        self.PROJECT_ROOT = root

        # ---- 路径 ----
        self.DATA_DIR = os.path.join(root, "data", "laws")
        self.MODEL_DIR = os.path.join(root, "models")
        self.VECTOR_DB_PATH = os.path.join(root, "vector_db")
        self.LOG_DIR = os.path.join(root, "logs")

        self.DATABASE_PATH = self.DATA_DIR  # 兼容旧名

        for d in [self.DATA_DIR, self.MODEL_DIR, self.VECTOR_DB_PATH, self.LOG_DIR]:
            os.makedirs(d, exist_ok=True)

        # ---- LLM ----
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
        self.LLM_API_KEY = os.getenv("LLM_API_KEY", "")

        if not self.LLM_API_KEY:
            raise ValueError(
                "未设置 LLM_API_KEY。请在 .env 中配置。\n"
                "示例: LLM_API_KEY=sk-xxx\n"
                "      LLM_PROVIDER=deepseek"
            )

        provider = LLM_PROVIDERS.get(self.LLM_PROVIDER)
        if not provider:
            raise ValueError(
                f"不支持的 LLM 厂商: '{self.LLM_PROVIDER}'。"
                f"可选: {', '.join(LLM_PROVIDERS.keys())}"
            )

        self.LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or provider["base_url"]
        self.LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or provider["default_model"]
        self.LLM_PROVIDER_NAME = provider["name"]

        # ---- 嵌入模型 ----
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

        # ---- 文本分割 ----
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

        # ---- 检索 ----
        self.TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
        self.MAX_RESULTS = int(os.getenv("MAX_RESULTS", "4"))
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))
        self.RETRIEVAL_PER_QUERY_K = 4
        self.RETRIEVAL_ADAPTIVE = True
        self.RETRIEVAL_MAX_TOP_K = 10
        self.RETRIEVAL_MIN_UNIQUE_RESULTS = 4
