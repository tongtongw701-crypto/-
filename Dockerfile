# ============================================================
# 法律智能问答系统 - Docker 镜像
# ============================================================
FROM python:3.11-slim

# 系统依赖 (编译某些 Python 包需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖清单，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建必要目录
RUN mkdir -p logs vector_db models

# Streamlit 配置 (容器内禁用浏览器自动打开 + 监听所有地址)
RUN mkdir -p ~/.streamlit && \
    echo "[server]\nheadless = true\naddress = \"0.0.0.0\"\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\n" > ~/.streamlit/config.toml && \
    echo "[theme]\nprimaryColor = \"#2196F3\"\nbackgroundColor = \"#FFFFFF\"\nsecondaryBackgroundColor = \"#F0F2F6\"\n" >> ~/.streamlit/config.toml

EXPOSE 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 默认启动 Streamlit UI
CMD ["python", "-m", "streamlit", "run", "src/ui/streamlit_app.py"]
