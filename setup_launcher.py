#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法律智能问答系统 - 启动前环境检查与配置
由 启动法律问答.bat 调用，负责虚拟环境、依赖安装、配置检查
"""

import os
import sys
import re
import subprocess
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.absolute()


def print_step(msg: str):
    """打印步骤标题"""
    print(f"  {msg}")


def print_ok(msg: str = ""):
    """打印成功信息"""
    print(f"    [OK] {msg}")


def print_warn(msg: str):
    """打印警告"""
    print(f"    [!] {msg}")


def print_fail(msg: str):
    """打印失败"""
    print(f"    [FAIL] {msg}")


def find_python() -> Optional[str]:
    """找到可用的 Python"""
    # 先尝试 venv 中的
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)

    # 再尝试系统 Python
    try:
        result = subprocess.run(
            [sys.executable, "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return sys.executable
    except Exception:
        pass

    # 尝试 where python
    try:
        result = subprocess.run(
            ["where", "python"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0].strip()
    except Exception:
        pass

    return None


def setup_venv() -> str:
    """创建/使用虚拟环境，返回 python 路径"""
    venv_dir = PROJECT_ROOT / ".venv"
    venv_python = venv_dir / "Scripts" / "python.exe"

    print_step("Checking virtual environment...")

    if venv_python.exists():
        print_ok("Virtual environment found")
        return str(venv_python)

    print("    Creating virtual environment...")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True, capture_output=True, text=True, timeout=120
        )
        print_ok("Virtual environment created")
        return str(venv_python)
    except subprocess.CalledProcessError as e:
        print_fail(f"Cannot create venv: {e.stderr}")
        print_warn("Will use system Python instead")
        return sys.executable
    except Exception as e:
        print_fail(f"Cannot create venv: {e}")
        print_warn("Will use system Python instead")
        return sys.executable


def check_dependency(python_path: str, module: str) -> bool:
    """检查单个模块是否可导入"""
    try:
        result = subprocess.run(
            [python_path, "-c", f"import {module}"],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except Exception:
        return False


def install_dependencies(python_path: str) -> bool:
    """安装项目依赖"""
    requirements = PROJECT_ROOT / "requirements.txt"

    if not requirements.exists():
        print_fail("requirements.txt not found")
        return False

    print_step("Checking dependencies...")

    # 先检查关键依赖
    critical_modules = ["streamlit", "langchain", "chromadb", "sentence_transformers"]
    all_ok = True
    for mod in critical_modules:
        if check_dependency(python_path, mod):
            print_ok(f"{mod}")
        else:
            print(f"    [MISS] {mod}")
            all_ok = False

    if all_ok:
        print_ok("All dependencies ready")
        fix_torch_on_windows(python_path)
        return True

    # 需要安装
    print()
    print_step("Installing dependencies...")
    print("    This may take 5-15 minutes on first run.")
    print("    Packages include torch (~2GB), chromadb, streamlit, etc.")
    print()

    # 使用 pip 安装，但不静默（让用户看到进度）
    # 设置 UTF-8 编码避免 Windows GBK 解码问题
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [python_path, "-m", "pip", "install", "-r", str(requirements)],
            check=False,
            timeout=1800,  # 30 分钟超时
            env=env,
        )
        if result.returncode == 0:
            print()
            print_ok("Dependencies installed successfully")
            fix_torch_on_windows(python_path)
            return True
        else:
            print()
            print_fail(f"Dependency installation failed (exit code: {result.returncode})")
            print()
            print("    Troubleshooting tips:")
            print("    1. Check your network connection")
            print("    2. Try: pip install -r requirements.txt")
            print("    3. For Chinese users, try mirror:")
            print("       pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
            return False
    except subprocess.TimeoutExpired:
        print_fail("Dependency installation timed out (30 min)")
        return False
    except Exception as e:
        print_fail(f"Installation error: {e}")
        return False


def fix_torch_on_windows(python_path: str) -> None:
    """Windows 上 PyTorch 常因 Anaconda DLL 冲突或 CUDA 版本问题加载失败。
    检测并重装为 CPU 版本（做文本嵌入不需要 GPU）。"""
    if sys.platform != "win32":
        return

    print_step("Checking torch compatibility...")
    try:
        result = subprocess.run(
            [python_path, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print_ok(f"torch {result.stdout.strip()}")
            return
    except Exception:
        pass

    print_warn("torch failed to load, reinstalling CPU-only version...")
    print("    This is a common Windows + Anaconda issue.")

    pip_args = [
        python_path, "-m", "pip", "install", "torch",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "--force-reinstall", "--no-deps",
    ]

    # 如果官方源慢，尝试清华镜像
    for extra_args in ([], ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]):
        try:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            result = subprocess.run(
                pip_args + extra_args,
                check=False, timeout=600, env=env,
            )
            if result.returncode == 0:
                print_ok("torch CPU version installed")
                return
        except subprocess.TimeoutExpired:
            continue

    print_warn("torch installation timed out; the app may not work for local search")


def check_configured() -> bool:
    """检查 .env 是否已配置"""
    env_file = PROJECT_ROOT / ".env"

    if not env_file.exists():
        return False

    try:
        content = env_file.read_text(encoding="utf-8")
        if "LLM_API_KEY=" not in content:
            return False

        match = re.search(r"LLM_API_KEY=(.+)", content)
        if not match:
            return False

        api_key = match.group(1).strip()
        if not api_key or "your_api_key_here" in api_key.lower() or api_key in ["your_api_key_here", "sk-xxx"]:
            return False

        return True
    except Exception:
        return False


def run_wizard(python_path: str) -> bool:
    """运行首次配置向导"""
    wizard = PROJECT_ROOT / "first_run_wizard.py"
    if not wizard.exists():
        print_fail("Setup wizard not found (first_run_wizard.py)")
        return False

    print()
    print("    Starting setup wizard...")
    print("    " + "=" * 46)
    print()

    try:
        result = subprocess.run(
            [python_path, str(wizard)],
            check=False
        )
        return result.returncode == 0
    except Exception as e:
        print_fail(f"Wizard error: {e}")
        return False


def check_vector_db() -> bool:
    """检查向量数据库"""
    vdb = PROJECT_ROOT / "vector_db"
    if not vdb.exists():
        return False
    return any(vdb.glob("*.sqlite3"))


def init_vector_db(python_path: str) -> bool:
    """初始化向量数据库"""
    init_script = PROJECT_ROOT / "init_vector_db.py"
    if not init_script.exists():
        print_fail("init_vector_db.py not found")
        return False

    print_step("Initializing vector database...")
    print("    This may take 2-5 minutes...")

    try:
        result = subprocess.run(
            [python_path, str(init_script)],
            check=False, timeout=600
        )
        if result.returncode == 0:
            print_ok("Vector database initialized")
            return True
        else:
            print_fail("Vector database initialization failed")
            return False
    except subprocess.TimeoutExpired:
        print_fail("Vector database initialization timed out")
        return False
    except Exception as e:
        print_fail(f"Error: {e}")
        return False


def main():
    """主流程"""
    print("=" * 50)
    print("  Legal QA System - Environment Setup")
    print("=" * 50)
    print()

    # 1. 找 Python / 创建 venv
    python_path = setup_venv()
    print()

    # 2. 安装依赖
    deps_ok = install_dependencies(python_path)
    if not deps_ok:
        print()
        print("[FAIL] Cannot continue without dependencies.")
        return 1
    print()

    # 3. 检查配置
    print_step("Checking configuration...")
    if not check_configured():
        print_warn("API Key not configured")
        if not run_wizard(python_path):
            print()
            print("[FAIL] Setup not complete.")
            return 1
        print()
        print_ok("Configuration complete!")
    else:
        print_ok("Configuration ready")
    print()

    # 4. 检查向量库
    print_step("Checking vector database...")
    if check_vector_db():
        print_ok("Vector database ready")
    else:
        print_warn("Vector database not initialized")
        print()
        answer = input("    Initialize now? (y/n, default y): ").strip().lower()
        if answer in ("", "y", "yes"):
            if not init_vector_db(python_path):
                print_warn("You can run 'python init_vector_db.py' manually later")
        else:
            print_warn("Local search will not work without vector database")

    print()
    print("=" * 50)
    print("  [OK] Setup complete! Launching...")
    print("=" * 50)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
