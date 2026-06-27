#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法律智能问答系统 - 首次运行配置向导
引导用户完成: API Key 配置 → 模型下载 → 向量库初始化
"""

import os
import sys
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║                                                      ║
║       ⚖️  法律智能问答系统 - 首次运行配置向导  ⚖️       ║
║                                                      ║
║     本向导将帮助您完成以下配置：                       ║
║       ① 选择 LLM 厂商并配置 API Key                  ║
║       ② 下载/加载嵌入模型                            ║
║       ③ 初始化法律知识向量库                         ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")


def step1_configure_llm():
    """步骤1: 配置 LLM API Key"""
    print("━" * 54)
    print("  ①  配置 LLM 大模型")
    print("━" * 54)
    print()
    print("  本系统支持以下 LLM 厂商（均兼容 OpenAI 接口）：")
    print()
    print("    1. DeepSeek     — deepseek.com       (推荐，便宜好用)")
    print("    2. 阿里云百炼    — dashscope.aliyun.com")
    print("    3. 月之暗面 Kimi — moonshot.cn")
    print("    4. 智谱 AI GLM  — open.bigmodel.cn")
    print("    5. OpenAI       — api.openai.com")
    print("    6. 自定义       — 自行输入 Base URL")
    print()

    provider_map = {
        "1": ("deepseek", "https://api.deepseek.com/v1", "deepseek-chat"),
        "2": ("dashscope", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-max"),
        "3": ("moonshot", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
        "4": ("zhipu", "https://open.bigmodel.cn/api/paas/v4", "glm-4"),
        "5": ("openai", "https://api.openai.com/v1", "gpt-4o-mini"),
    }

    while True:
        choice = input("  请选择厂商 (1-6): ").strip()
        if choice in provider_map:
            provider, base_url, default_model = provider_map[choice]
            break
        elif choice == "6":
            provider = "custom"
            base_url = input("  请输入 API Base URL: ").strip()
            if not base_url:
                print("  [!] Base URL 不能为空")
                continue
            default_model = input("  请输入默认模型名称: ").strip() or "default"
            break
        else:
            print("  [!] 请输入 1-6 之间的数字")

    print()
    api_key = input("  请输入 API Key: ").strip()
    while not api_key:
        print("  [!] API Key 不能为空")
        api_key = input("  请输入 API Key: ").strip()

    model = input(f"  模型名称 (回车使用默认: {default_model}): ").strip()
    if not model:
        model = default_model

    # 写入 .env 文件
    env_content = f"""# ============================================================
# 法律智能问答系统 - 环境变量配置
# 由首次运行向导自动生成
# ============================================================

# LLM 配置
LLM_API_KEY={api_key}
LLM_PROVIDER={provider}
LLM_BASE_URL={base_url}
LLM_MODEL={model}

# 嵌入模型 (首次运行会自动下载，约 1.3GB)
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

# 检索参数
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=4
SIMILARITY_THRESHOLD=0.4
"""

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(env_content)

    print()
    print(f"  [✓] API Key 已保存到 {ENV_FILE}")
    print()
    return True


def step2_download_model():
    """步骤2: 下载嵌入模型"""
    print("━" * 54)
    print("  ②  下载嵌入模型 (BGE-large-zh-v1.5)")
    print("━" * 54)
    print()
    print("  嵌入模型用于将法律文本转换为向量以进行语义检索。")
    print("  模型大小约 1.3GB，首次下载需要几分钟（仅需一次）。")
    print()

    model_dir = PROJECT_ROOT / "models" / "bge-large-zh-v1.5"
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"  [✓] 模型已存在: {model_dir}")
        print()
        return True

    print("  正在从 HuggingFace 下载模型...")
    print("  (如果下载缓慢，可设置 HF_MIRROR 环境变量使用镜像站)")
    print()

    try:
        from sentence_transformers import SentenceTransformer
        print("  下载中，请耐心等待...")
        model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
        # 保存到本地 models 目录
        model.save(str(model_dir))
        print(f"  [✓] 模型已下载到: {model_dir}")
        print()
        return True
    except Exception as e:
        print(f"  [!] 模型下载失败: {e}")
        print()
        print("  您可以尝试以下方法：")
        print("    1. 设置 HuggingFace 镜像: set HF_ENDPOINT=https://hf-mirror.com")
        print("    2. 手动下载模型放到 models/bge-large-zh-v1.5/ 目录")
        print("    3. 从以下地址下载: https://huggingface.co/BAAI/bge-large-zh-v1.5")
        print()
        retry = input("  下载失败，是否跳过此步骤？(y/n, 默认 n): ").strip().lower()
        if retry == "y":
            print("  [!] 已跳过模型下载，系统将在首次运行时自动尝试下载")
            return True
        return False


def step3_init_vector_db():
    """步骤3: 初始化向量数据库"""
    print("━" * 54)
    print("  ③  初始化法律知识向量库")
    print("━" * 54)
    print()
    print("  正在将法律文档转换为向量索引...")
    print("  这可能需要 2-5 分钟（仅需一次）。")
    print()

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "init_vector_db.py")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            print(result.stdout)
            print("  [✓] 向量库初始化完成！")
            print()
            return True
        else:
            print(result.stdout)
            print(result.stderr)
            print("  [!] 向量库初始化失败，请检查上方错误信息")
            return False
    except subprocess.TimeoutExpired:
        print("  [!] 初始化超时（超过 10 分钟），请手动运行: python init_vector_db.py")
        return False
    except Exception as e:
        print(f"  [!] 初始化出错: {e}")
        return False


def check_dependencies():
    """检查依赖是否安装"""
    print("━" * 54)
    print("  检查 Python 依赖...")
    print("━" * 54)
    print()

    missing = []
    required = [
        ("langchain", "langchain>=0.2.0"),
        ("chromadb", "chromadb>=0.4.0"),
        ("sentence_transformers", "sentence-transformers>=2.2.0"),
        ("streamlit", "streamlit>=1.30.0"),
        ("requests", "requests>=2.28.0"),
        ("dotenv", "python-dotenv>=1.0.0"),
        ("bs4", "beautifulsoup4>=4.12.0"),
        ("docx", "python-docx>=0.8.11"),
    ]

    for module, pkg in required:
        try:
            __import__(module)
            print(f"  [✓] {pkg}")
        except ImportError:
            print(f"  [✗] {pkg} — 未安装")
            missing.append(pkg)

    if missing:
        print()
        print(f"  [!] 缺少 {len(missing)} 个依赖")
        install = input("  是否自动安装？(y/n, 默认 y): ").strip().lower()
        if install != "n":
            print("  正在安装依赖...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r",
                 str(PROJECT_ROOT / "requirements.txt")]
            )
            print("  [✓] 依赖安装完成！")
            print()
            return True
        else:
            print(f"  请手动运行: pip install -r requirements.txt")
            return False

    print()
    print("  [✓] 所有依赖已就绪！")
    print()
    return True


def check_already_configured():
    """检查是否已经配置过"""
    if ENV_FILE.exists():
        content = ENV_FILE.read_text(encoding="utf-8")
        if "LLM_API_KEY=" in content and "your_api_key_here" not in content:
            api_key_match = re.search(r"LLM_API_KEY=(.+)", content)
            if api_key_match and len(api_key_match.group(1).strip()) > 5:
                return True
    return False


def main():
    """主流程"""
    print_banner()

    # 检查依赖
    if not check_dependencies():
        print("请先安装依赖后再运行此向导。")
        input("按回车键退出...")
        return 1

    # 检查是否已配置
    if check_already_configured():
        print("  检测到已有配置文件 .env")
        reconfigure = input("  是否重新配置？(y/n, 默认 n): ").strip().lower()
        if reconfigure != "y":
            print("  跳过配置，使用现有设置。")
            print()
        else:
            step1_configure_llm()
    else:
        step1_configure_llm()

    # 下载模型
    model_done = step2_download_model()
    if not model_done:
        print("[!] 模型下载失败，系统可能无法正常运行")
        cont = input("是否继续？(y/n, 默认 y): ").strip().lower()
        if cont == "n":
            return 1

    # 初始化向量库
    vector_db_path = PROJECT_ROOT / "vector_db"
    if vector_db_path.exists() and list(vector_db_path.glob("*.sqlite3")):
        print("━" * 54)
        print("  ③  向量库已存在，跳过初始化")
        print("━" * 54)
        print(f"  [✓] 向量库路径: {vector_db_path}")
        print()
    else:
        if not step3_init_vector_db():
            print("[!] 向量库初始化失败")
            cont = input("是否继续？(y/n, 默认 y): ").strip().lower()
            if cont == "n":
                return 1

    # 完成
    print("╔══════════════════════════════════════════════════════╗")
    print("║                                                      ║")
    print("║           🎉  配置完成！系统已就绪！                  ║")
    print("║                                                      ║")
    print("║  启动方式:                                           ║")
    print("║    • Web 界面: python run_ui.py                      ║")
    print("║    • 命令行:   python run_qa_system.py               ║")
    print("║    • Docker:   docker compose up -d                  ║")
    print("║                                                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    launch = input("是否立即启动 Web 界面？(y/n, 默认 y): ").strip().lower()
    if launch != "n":
        print("正在启动...")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "run_ui.py")])

    return 0


if __name__ == "__main__":
    sys.exit(main())
