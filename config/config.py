#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件
包含所有系统配置项，支持多家 LLM 厂商
"""

import os
from pathlib import Path

# 自动加载 .env 文件 (如果存在)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv 未安装时忽略


# ============================================================
# LLM 厂商预设配置
# 大部分国产大模型均兼容 OpenAI 接口，通过 base_url 切换即可
# ============================================================
LLM_PROVIDERS = {
    "dashscope": {
        "name": "阿里云 DashScope (通义千问)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-max",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "xiaomi": {
        "name": "小米 MiLM",
        "base_url": "https://api.xiaomi.com/v1",
        "default_model": "MiLM-6B",
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
    """配置类"""

    def __init__(self):
        # 项目根目录
        self.PROJECT_ROOT = Path(__file__).parent.parent.absolute()

        # 法律文档目录 (优先 data/laws，兼容旧版 Database/)
        data_laws_dir = os.path.join(self.PROJECT_ROOT, "data", "laws")
        legacy_dir = os.path.join(self.PROJECT_ROOT, "Database")
        if os.path.isdir(data_laws_dir) and os.listdir(data_laws_dir):
            self.DATA_DIR = data_laws_dir
        elif os.path.isdir(legacy_dir) and os.listdir(legacy_dir):
            self.DATA_DIR = legacy_dir
        else:
            self.DATA_DIR = data_laws_dir
        self.DATABASE_PATH = self.DATA_DIR

        # 模型目录
        self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, "models")

        # 向量数据库目录
        self.VECTOR_DB_PATH = os.path.join(self.PROJECT_ROOT, "vector_db")

        # 日志目录
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, "logs")

        # 创建必要的目录
        self._create_directories()

        # --- LLM 配置 ---
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
        self.LLM_API_KEY = os.getenv("LLM_API_KEY", "")
        # 兼容旧版环境变量名
        if not self.LLM_API_KEY:
            self.LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

        if not self.LLM_API_KEY:
            _providers = " / ".join(LLM_PROVIDERS.keys())
            raise ValueError(
                "未设置 LLM_API_KEY 环境变量。\n"
                f"支持的厂商: {_providers}\n"
                "请在 .env 文件或系统环境变量中配置。\n"
                "示例: LLM_API_KEY=sk-xxx\n"
                "      LLM_PROVIDER=deepseek"
            )

        # 解析厂商配置
        provider_cfg = LLM_PROVIDERS.get(self.LLM_PROVIDER)
        if not provider_cfg:
            _valid = ", ".join(LLM_PROVIDERS.keys())
            raise ValueError(
                f"不支持的 LLM 厂商: '{self.LLM_PROVIDER}'\n"
                f"可选值: {_valid}"
            )

        self.LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or provider_cfg["base_url"]
        self.LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or provider_cfg["default_model"]
        self.LLM_PROVIDER_NAME = provider_cfg["name"]

        # 嵌入模型配置
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

        # 文本分割配置
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

        # 检索配置
        self.TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
        self.MAX_RESULTS = int(os.getenv("MAX_RESULTS", "4"))
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))

        # 检索优化配置
        self.RETRIEVAL_PER_QUERY_K = 4
        self.RETRIEVAL_ADAPTIVE = True
        self.RETRIEVAL_MAX_TOP_K = 10
        self.RETRIEVAL_MIN_UNIQUE_RESULTS = 4

    def _create_directories(self):
        """创建必要的目录"""
        for d in [self.DATA_DIR, self.MODEL_DIR, self.VECTOR_DB_PATH, self.LOG_DIR]:
            os.makedirs(d, exist_ok=True)

    def get_data_file_path(self, filename: str) -> str:
        """获取数据文件的完整路径"""
        return os.path.join(self.DATA_DIR, filename)
