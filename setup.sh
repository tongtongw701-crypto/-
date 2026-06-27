#!/usr/bin/env bash
# ============================================================
# 法律智能问答系统 - macOS/Linux 一键启动脚本
# 使用方式: bash setup.sh
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       ⚖️  法律智能问答系统 - 一键启动  ⚖️             ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# --- 1. 检查 Python ---
echo -e "${CYAN}[1/4]${NC} 检查 Python 环境..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}[✗] 未找到 Python！${NC}"
    echo ""
    echo "请先安装 Python 3.9 或更高版本:"
    echo "  macOS:  brew install python@3.11"
    echo "  Ubuntu: sudo apt install python3 python3-venv python3-pip"
    echo ""
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo -e "  ${GREEN}[✓]${NC} $PYTHON_VERSION"

# --- 2. 检查/创建虚拟环境 ---
echo ""
echo -e "${CYAN}[2/4]${NC} 检查虚拟环境..."

VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "  正在创建虚拟环境..."
    $PYTHON -m venv "$VENV_DIR"
    echo -e "  ${GREEN}[✓]${NC} 虚拟环境已创建"
else
    echo -e "  ${GREEN}[✓]${NC} 虚拟环境已存在"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# --- 3. 安装/更新依赖 ---
echo ""
echo -e "${CYAN}[3/4]${NC} 检查依赖..."

if python -c "import streamlit, langchain, chromadb, sentence_transformers" &> /dev/null; then
    echo -e "  ${GREEN}[✓]${NC} 依赖已就绪"
else
    echo "  正在安装依赖（首次可能需要几分钟）..."
    pip install -r requirements.txt -q
    echo -e "  ${GREEN}[✓]${NC} 依赖安装完成"
fi

# --- 4. 检查配置 ---
echo ""
echo -e "${CYAN}[4/4]${NC} 检查系统配置..."

if [ ! -f ".env" ]; then
    echo -e "  ${YELLOW}[!]${NC} 未检测到配置文件"
    echo ""
    echo "  正在启动配置向导..."
    echo "  ═══════════════════════════════════════════════════"
    python first_run_wizard.py
    echo "  ═══════════════════════════════════════════════════"
    echo ""
    echo -e "  ${GREEN}[✓]${NC} 配置完成！"
else
    echo -e "  ${GREEN}[✓]${NC} 配置已就绪"
fi

# 检查向量库
if [ ! -f "vector_db/chroma.sqlite3" ]; then
    echo ""
    echo -e "  ${YELLOW}[!]${NC} 向量数据库尚未初始化"
    read -p "  是否立即初始化? (y/n, 默认 y): " init_choice
    init_choice=${init_choice:-y}
    if [ "$init_choice" = "y" ]; then
        echo "  正在初始化向量数据库（需要几分钟）..."
        python init_vector_db.py
    fi
fi

# --- 启动 ---
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              🚀  正在启动系统...                     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  浏览器将自动打开，如未打开请访问:"
echo "  http://localhost:8501"
echo ""
echo "  按 Ctrl+C 可停止服务"
echo ""

# 尝试自动打开浏览器
if command -v open &> /dev/null; then
    open http://localhost:8501 2>/dev/null &
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8501 2>/dev/null &
fi

python -m streamlit run src/ui/streamlit_app.py --server.port 8501 --server.headless true
