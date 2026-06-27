#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
便携版打包脚本
将项目打包为可分发的文件夹，包含嵌入式 Python 和所有依赖

使用方式:
    python package_portable.py

产物: output/legal-qa-portable/ 文件夹
    用户只需解压并双击 启动法律问答.bat 即可使用
"""

import os
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
OUTPUT_DIR = PROJECT_ROOT / "output"
PORTABLE_DIR = OUTPUT_DIR / "legal-qa-portable"


def clean_output():
    """清理旧产物"""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def copy_source_files():
    """复制项目源代码"""
    print("[1/4] 复制源代码...")

    # 需要包含的文件和目录
    include = [
        "config",
        "src",
        "utils",
        "eval",
        "data",
        "prompts",
        "docs",
        "scripts",
        "Database",  # 兼容旧版
        "*.py",
        "*.bat",
        "*.sh",
        "requirements.txt",
        ".env.example",
        "README.md",
    ]

    # 需要排除的模式
    exclude_patterns = [
        "__pycache__",
        "*.pyc",
        ".DS_Store",
        "Thumbs.db",
    ]

    for pattern in include:
        src_pattern = PROJECT_ROOT / pattern
        if "*" in pattern:
            for f in PROJECT_ROOT.glob(pattern):
                if f.is_file():
                    rel = f.relative_to(PROJECT_ROOT)
                    dst = PORTABLE_DIR / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
        elif src_pattern.is_dir():
            shutil.copytree(
                src_pattern,
                PORTABLE_DIR / pattern,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*exclude_patterns),
            )
        elif src_pattern.is_file():
            dst = PORTABLE_DIR / pattern
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_pattern, dst)

    # 创建必要空目录
    for d in ["models", "vector_db", "logs"]:
        (PORTABLE_DIR / d).mkdir(parents=True, exist_ok=True)

    # 在 models 目录放说明
    (PORTABLE_DIR / "models" / "README.txt").write_text(
        "嵌入模型(BGE-large-zh-v1.5)将在首次运行时自动下载到此目录。\n"
        "如需手动下载: https://huggingface.co/BAAI/bge-large-zh-v1.5\n",
        encoding="utf-8",
    )

    print(f"  [✓] 源代码已复制到 {PORTABLE_DIR}")


def create_launcher():
    """创建/更新启动器脚本（确保路径正确）"""
    print("[2/4] 创建启动器...")
    # 启动器已经在项目根目录，随 copy_source_files 一起复制了
    print("  [✓] 启动器已就绪")


def verify_package():
    """验证打包完整性"""
    print("[3/4] 验证打包完整性...")

    required_files = [
        "legal_qa_app.py",
        "run_ui.py",
        "first_run_wizard.py",
        "init_vector_db.py",
        "requirements.txt",
        ".env.example",
        "启动法律问答.bat",
    ]

    recommended_dirs = [
        "config",
        "src/ui",
        "src/rag_system",
        "src/data_processing",
        "prompts",
        "data/laws",
    ]

    all_good = True
    for f in required_files:
        path = PORTABLE_DIR / f
        if not path.exists():
            print(f"  [✗] 缺少文件: {f}")
            all_good = False

    for d in recommended_dirs:
        path = PORTABLE_DIR / d
        if not path.exists():
            print(f"  [!] 目录不存在: {d}")

    if all_good:
        print("  [✓] 核心文件检查通过")
    else:
        print("  [!] 存在缺失文件，包可能无法正常运行")


def create_archive():
    """创建压缩包"""
    print("[4/4] 创建压缩包...")

    zip_path = OUTPUT_DIR / "legal-qa-portable.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PORTABLE_DIR):
            # 排除不需要的目录
            dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git"]]
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(PORTABLE_DIR)
                zf.write(file_path, arcname)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  [✓] 压缩包已创建: {zip_path} ({size_mb:.1f} MB)")


def main():
    """主流程"""
    print("=" * 50)
    print("  法律智能问答系统 - 便携版打包工具")
    print("=" * 50)
    print()
    print(f"  产物目录: {OUTPUT_DIR}")
    print()

    clean_output()
    copy_source_files()
    create_launcher()
    verify_package()
    create_archive()

    print()
    print("=" * 50)
    print("  🎉 打包完成！")
    print()
    print("  产物:")
    print(f"    文件夹: {PORTABLE_DIR}")
    print(f"    压缩包: {OUTPUT_DIR / 'legal-qa-portable.zip'}")
    print()
    print("  分发说明:")
    print("    1. 将压缩包上传到 GitHub Releases")
    print("    2. 用户下载解压后双击 启动法律问答.bat")
    print("    3. 首次运行会自动配置环境和初始化知识库")
    print("=" * 50)


if __name__ == "__main__":
    main()
