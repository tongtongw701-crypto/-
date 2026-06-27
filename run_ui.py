#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法律智能问答系统 - Streamlit UI 启动器
自动检测配置状态，未配置时引导至首次运行向导
"""

import sys
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.absolute()
ENV_FILE = PROJECT_ROOT / ".env"


def check_configured() -> bool:
    """快速检查系统是否已完成基本配置"""
    if not ENV_FILE.exists():
        return False

    try:
        content = ENV_FILE.read_text(encoding="utf-8")
        if "LLM_API_KEY=" not in content:
            return False
        match = re.search(r"LLM_API_KEY=(.+)", content)
        if not match:
            return False
        api_key = match.group(1).strip()
        if not api_key or "your_api_key_here" in api_key.lower():
            return False
        return True
    except Exception:
        return False


def check_vector_db() -> bool:
    """检查向量数据库是否已初始化"""
    vdb = PROJECT_ROOT / "vector_db"
    if not vdb.exists():
        return False
    return any(vdb.glob("*.sqlite3"))


def run_first_run_wizard():
    """启动配置向导"""
    wizard_path = PROJECT_ROOT / "first_run_wizard.py"
    if wizard_path.exists():
        print("正在启动首次运行配置向导...")
        print("=" * 50)
        result = subprocess.run([sys.executable, str(wizard_path)])
        return result.returncode == 0
    else:
        print("[X] 未找到配置向导文件 first_run_wizard.py")
        print("请手动创建 .env 文件并配置 LLM_API_KEY")
        return False


def run_streamlit():
    """启动 Streamlit UI 界面"""
    ui_path = PROJECT_ROOT / "src" / "ui" / "streamlit_app.py"

    if not ui_path.exists():
        print(f"错误: 找不到 UI 文件 {ui_path}")
        return False

    print("正在启动法律智能问答可视化界面...")
    print("如果浏览器没有自动打开，请手动访问: http://localhost:8501")
    print()

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(ui_path),
            "--server.port", "8501",
            "--server.headless", "true",
        ])
        return True
    except KeyboardInterrupt:
        print("\n界面已关闭")
        return True
    except Exception as e:
        print(f"启动失败: {e}")
        print("\n请确保已安装 streamlit: pip install streamlit")
        return False


def main():
    """主流程"""
    print("=" * 50)
    print("  ⚖️  法律智能问答系统 - Web 界面启动器")
    print("=" * 50)
    print()

    # 1. 检查配置
    if not check_configured():
        print("[!] 系统尚未配置（缺少 API Key）")
        print()
        print("即将启动配置向导...")
        print()
        if not run_first_run_wizard():
            print("\n[!] 配置未完成，无法启动系统。")
            print("请重新运行本程序完成配置。")
            input("按回车键退出...")
            return 1

    # 2. 检查向量库
    if not check_vector_db():
        print()
        print("[!] 向量数据库尚未初始化")
        choice = input("是否立即初始化？(y/n, 默认 y): ").strip().lower()
        if choice != "n":
            print("正在初始化向量数据库（需要几分钟）...")
            init_script = PROJECT_ROOT / "init_vector_db.py"
            if init_script.exists():
                result = subprocess.run(
                    [sys.executable, str(init_script)],
                    cwd=str(PROJECT_ROOT),
                )
                if result.returncode != 0:
                    print("[!] 向量库初始化失败，系统可能无法正常使用本地检索功能")
            else:
                print("[X] 未找到初始化脚本 init_vector_db.py")

    # 3. 启动 Streamlit
    print()
    run_streamlit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
