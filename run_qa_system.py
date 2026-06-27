#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法律问答系统启动脚本
"""

import os
import sys
import subprocess
import platform

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import langchain
        import chromadb
        import sentence_transformers
        import requests
        print("[OK] 所有依赖检查通过")
        return True
    except ImportError as e:
        print(f"[X] 缺少必要依赖: {e}")
        return False

def check_vector_db_initialized(vector_db_path):
    """检查向量数据库是否已初始化"""
    if not os.path.exists(vector_db_path):
        return False
    
    # 检查数据库目录是否包含必要的文件
    required_files = ["chroma.sqlite3"]
    for file in required_files:
        if not os.path.exists(os.path.join(vector_db_path, file)):
            return False
    
    return True

def setup_environment():
    """设置运行环境"""
    # 设置工作目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # 添加项目根目录到Python路径
    sys.path.insert(0, project_root)
    
    print(f"[OK] 工作目录设置为: {project_root}")

def main():
    """主函数"""
    print("=" * 60)
    print("智能法律咨询助手启动程序")
    print("=" * 60)
    
    # 设置环境
    setup_environment()
    
    # 检查依赖
    if not check_dependencies():
        print("请先安装必要的依赖:")
        print("pip install -r requirements.txt")
        return 1
    
    # 检查向量数据库是否已初始化
    from config.config import Config
    config = Config()
    
    if not check_vector_db_initialized(config.VECTOR_DB_PATH):
        print("[!] 向量数据库尚未初始化或初始化不完整")
        print("请先运行初始化脚本:")
        print("python init_vector_db.py")
        return 1
    
    # 导入并运行主应用
    try:
        from legal_qa_app import LegalAgent
        app = LegalAgent()
        app.run()
    except Exception as e:
        print(f"[X] 启动应用时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())