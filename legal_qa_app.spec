# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置文件
用于将法律智能问答系统打包为独立可执行程序

使用方式:
    pip install pyinstaller
    pyinstaller legal_qa_app.spec

输出: dist/法律智能问答/ 文件夹，内含 法律智能问答.exe

⚠️ 注意:
  - 打包后体积约 2-3GB (含 torch, chromadb, streamlit 等)
  - 嵌入模型和向量库需要在首次运行时配置
  - API Key 由用户自行配置
"""

import os
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# ---- 收集所有 Python 源文件和数据文件 ----

# 需要打包的 Python 包
hidden_imports = [
    # Streamlit 及其依赖
    "streamlit",
    "streamlit.runtime",
    "streamlit.runtime.scriptrunner",
    "streamlit.web",
    "streamlit.web.server",
    "streamlit.elements",
    "streamlit.commands",
    "streamlit.components",
    "altair",
    "pandas",
    "numpy",
    "pyarrow",
    "pillow",
    "rich",
    "toml",
    "watchdog",
    "gitpython",
    "pydeck",
    "blinker",
    "tzlocal",
    "validators",
    "cachetools",

    # LangChain 及其依赖
    "langchain",
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_community.embeddings",
    "langchain.text_splitter",
    "langchain.docstore",
    "langchain.schema",
    "tiktoken",
    "yaml",
    "jsonpatch",

    # ChromaDB
    "chromadb",
    "chromadb.config",
    "chromadb.api",
    "chromadb.db",
    "chromadb.telemetry",
    "pydantic",
    "orjson",
    "hnswlib",
    "sqlite3",
    "importlib_resources",

    # Sentence Transformers + Transformers + Torch
    "sentence_transformers",
    "transformers",
    "transformers.models",
    "tokenizers",
    "huggingface_hub",
    "torch",
    "torchvision",
    "scipy",
    "sklearn",
    "numpy",
    "onnxruntime",
    "protobuf",

    # 网络/解析
    "requests",
    "urllib3",
    "chardet",
    "certifi",
    "idna",
    "beautifulsoup4",
    "bs4",
    "lxml",
    "html.parser",

    # 文档处理
    "docx",
    "python-docx",
    "docx2txt",

    # 中文分词
    "jieba",

    # 其他
    "python-dotenv",
    "dotenv",
    "re",
    "json",
    "uuid",
    "datetime",
    "pathlib",
    "concurrent.futures",
    "asyncio",
    "multiprocessing",
    "logging",
    "typing",
    "tqdm",
    "filelock",
    "regex",

    # 评估相关
    "datasets",
    "ragas",
    "langchain-openai",
]

# 需要包含的数据文件（相对于项目根目录）
datas = [
    # 提示词模板
    ("prompts/*.txt", "prompts"),
    ("prompts/__init__.py", "prompts"),

    # 法律文档（用于初始化向量库）
    ("data/laws/*.docx", "data/laws"),

    # 配置文件
    ("config/*.py", "config"),

    # Streamlit UI
    ("src/ui/*.py", "src/ui"),

    # RAG 系统
    ("src/rag_system/*.py", "src/rag_system"),

    # 数据处理
    ("src/data_processing/*.py", "src/data_processing"),

    # 法律分析
    ("src/legal_analysis/*.py", "src/legal_analysis"),

    # 网络资源
    ("src/online_resources/*.py", "src/online_resources"),

    # 工具
    ("utils/*.py", "utils"),

    # 评估
    ("eval/*.py", "eval"),
]

# 排除不必要的模块（减小体积）
excluded_modules = [
    "tkinter",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "nbformat",
    "nbconvert",
    "sympy",
    "sphinx",
    "pytest",
    "setuptools",
    "distutils",
    "pip",
    "wheel",
    "Cython",
    "llvmlite",
    "numba",
    "cupy",
    "tensorflow",
    "keras",
    "opencv",
    "cv2",
    "PIL.ImageQt",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "wx",
]

a = Analysis(
    ["legal_qa_app.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    noarchive=False,
)

# 手动添加数据文件
for src_pattern, dst in datas:
    import glob as _glob
    src_path = os.path.join(str(PROJECT_ROOT), src_pattern)
    for f in _glob.glob(src_path):
        if os.path.isfile(f):
            rel = os.path.relpath(f, str(PROJECT_ROOT))
            target_dir = os.path.join(dst, os.path.basename(f))
            a.datas.append((f, os.path.dirname(target_dir)))

pyz = PYZ(a.pure)

# 单文件 exe 或 单文件夹
# 注: 对于本项目，推荐使用单文件夹模式（启动更快，方便添加模型和向量库）
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="法律智能问答",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,   # 显示控制台（方便查看日志）
    icon=None,       # 可替换为自定义图标路径
    disable_windowed_tracked=False,
)

# 收集所有文件到 dist 文件夹
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="法律智能问答",
)
