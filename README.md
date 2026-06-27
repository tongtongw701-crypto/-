# ⚖️ 法律智能问答系统

基于 **RAG（检索增强生成）** 架构的中文法律智能问答系统，支持 7 部法律的法条检索、网络信息增强和赔偿金额计算。

<!-- TODO: 替换为实际截图 -->
<!-- ![界面截图](docs/images/screenshot.png) -->

## ✨ 核心特性

- **🔍 多查询扩展检索** — 自动提取法名/条号，生成多组检索 query，自适应 Rerank 降噪
- **🧠 两阶段意图路由** — 规则关键词匹配（快速）+ LLM 语义分析（精准），智能调度本地检索/网络搜索/计算器
- **🌐 网络搜索增强** — 百度搜索结果爬取，基于域名可信度评分（gov.cn > edu.cn > 一般站点）
- **🧮 法律赔偿计算器** — 从法条文本中自动提取利率/比例/倍数等计算公式，支持逾期利息、赔偿金等计算
- **💬 流式对话** — 多会话管理，支持上下文追问，流式逐字输出
- **📊 RAGAS 评估体系** — 140 题评估集 + faithfulness / answer_relevancy / context_precision / context_recall 四维指标

## 🏗️ 系统架构

```
用户问题
    │
    ▼
┌─────────────────────────────────┐
│  1. 预增强 (Query Rewriting)     │  ← LLM 改写追问、补充历史上下文
│  2. 意图路由 (Intent Router)     │  → 规则匹配 + LLM 语义分析
│  3. 查询增强 (Query Enhancement) │  → 针对工具优化搜索词
└─────────┬───────────┬───────────┘
          │           │
    ┌─────▼─────┐ ┌───▼───────┐
    │ 本地检索   │ │ 网络搜索   │
    │ (Chroma)  │ │ (Baidu)   │
    └─────┬─────┘ └───┬───────┘
          │           │
    ┌─────▼───────────▼───────┐
    │ 4. 聚合验证 (Aggregation) │  ← 交叉验证、可信度排序
    │ 5. 最终生成 (Generation)  │  ← 证据引用强制、禁止编造法条
    └──────────────────────────┘
          │
          ▼
      流式回答
```

## 📁 目录结构

```
legal-qa-system/
├── config/
│   └── config.py              # 集中配置 (环境变量驱动)
├── data/laws/                 # 法律文档 (.docx)
├── src/
│   ├── data_processing/       # 文档加载与切分
│   ├── legal_analysis/        # 法律领域分析
│   ├── rag_system/            # RAG 引擎与 Chroma 客户端
│   ├── online_resources/      # 网络搜索模块
│   └── ui/                    # Streamlit 前端
├── eval/                      # 评估数据集与分析脚本
├── scripts/                   # 初始化与评估脚本
├── tests/                     # 测试用例
├── legal_qa_app.py            # 核心 Agent
├── docker-compose.yml         # Docker 一键部署
└── requirements.txt           # Python 依赖
```

## 🚀 快速开始

### 方式一：一键启动（推荐 ⭐）

**下载 → 双击 → 完成！** 脚本自动处理环境配置、依赖安装、配置引导。

| 平台 | 操作 |
|------|------|
| **Windows** | 下载项目 → 双击 `启动法律问答.bat` |
| **macOS / Linux** | 下载项目 → 终端运行 `bash setup.sh` |

首次运行会自动启动**配置向导**，引导您：
1. 选择 LLM 厂商并填入 API Key（支持 DeepSeek / 通义千问 / Kimi / 智谱 / OpenAI）
2. 下载嵌入模型（自动，约 1.3GB，仅需一次）
3. 初始化法律知识向量库（约 2-5 分钟，仅需一次）

配置完成后浏览器自动打开 http://localhost:8501 🎉

> **前置要求**: Python >= 3.9。如果未安装，请先访问 [python.org](https://www.python.org/downloads/) 下载安装（安装时勾选 "Add Python to PATH"）。

---

### 方式二：手动配置运行

> 适用于需要精细控制环境的开发者。

**1. 环境要求**
- Python >= 3.9

**2. 安装依赖**

```bash
# 建议使用虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. 运行配置向导**

```bash
python first_run_wizard.py
```

**4. 启动界面**

```bash
python run_ui.py
# 浏览器打开 http://localhost:8501
```

---

### 方式三：命令行交互

```bash
python run_qa_system.py
```

---

### 方式四：Docker 部署

```bash
# 1. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY

# 2. 初始化向量库
docker compose run --rm app python init_vector_db.py

# 3. 启动服务
docker compose up -d

# 4. 访问 http://localhost:8501
```

---

### 方式五：打包为独立可执行程序（高级）

使用 PyInstaller 打包为 Windows .exe：

```bash
pip install pyinstaller
pyinstaller legal_qa_app.spec
# 产物在 dist/法律智能问答/ 目录
```

> ⚠️ 打包后体积约 2-3GB。嵌入模型和向量库仍需首次运行时配置。

---

## 📐 配置说明

所有配置集中在 [config/config.py](config/config.py)，支持环境变量覆盖：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DASHSCOPE_API_KEY` | (必填) | DashScope API 密钥 |
| `QWEN_MODEL` | `qwen-max` | 生成模型名称 |
| `EMBEDDING_MODEL` | `BAAI/bge-large-zh-v1.5` | 嵌入模型 |
| `CHUNK_SIZE` | `1000` | 文本切分块大小 |
| `CHUNK_OVERLAP` | `200` | 切分重叠字符数 |
| `TOP_K_RESULTS` | `4` | 检索返回结果数 |
| `SIMILARITY_THRESHOLD` | `0.4` | 相似度阈值 |
| `CHROMA_SERVER_URL` | (空) | 远程 Chroma Server 地址 |

## 📊 评估

本项目包含完整的 RAG 评估体系：

```bash
# 运行 RAGAS 评估 (需要先启动 Chroma Server 或使用本地语料)
python evaluate_rag.py

# 分析评估结果
python eval/analyze_eval.py
```

评估指标：
- **Faithfulness** — 回答是否忠于检索到的上下文
- **Answer Relevancy** — 回答与问题的相关性
- **Context Precision** — 检索结果的精确度
- **Context Recall** — 检索结果的召回率

## 📚 支持的法律

| 法律 | 文档 |
|------|------|
| 中华人民共和国宪法 | 2018 年修正文本 |
| 中华人民共和国民法典 | 2020 |
| 中华人民共和国刑法 | 2020 修正 |
| 中华人民共和国劳动法 | 2018 修正 |
| 中华人民共和国劳动合同法 | 2012 修正 |
| 中华人民共和国民事诉讼法 | 2023 修正 |
| 中华人民共和国公司法 | 2023 修订 |

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| LLM | 通义千问 (Qwen-Max via DashScope) |
| 嵌入模型 | BAAI/bge-large-zh-v1.5 |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB |
| 前端 | Streamlit |
| 评估 | RAGAS |
| 部署 | Docker / Docker Compose |
| 打包 | PyInstaller / portable |

## 📦 发布到 GitHub Releases

本项目已配置 GitHub Actions 自动构建。发布新版本只需：

```bash
# 1. 确保所有更改已提交
git add .
git commit -m "v1.0.0 发布准备"

# 2. 创建版本 tag 并推送
git tag v1.0.0
git push origin main --tags
```

GitHub Actions 将自动：
- 打包源代码为 .zip / .tar.gz
- 创建 GitHub Release 并上传产物
- 验证代码完整性

用户从 [Releases](https://github.com/你的用户名/legal-qa-system/releases) 页面下载后，解压双击即可使用。

> **手动打包**: 运行 `python package_portable.py` 可创建便携版压缩包。

---

## ⚠️ 免责声明

本系统仅用于技术演示和学习研究，AI 生成的回答**不构成法律建议**。如有实际法律问题，请咨询专业律师。

## 📄 许可证

[MIT License](LICENSE)
