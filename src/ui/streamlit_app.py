import streamlit as st
import os
import sys
import uuid
from datetime import datetime

# 将项目根目录添加到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from legal_qa_app import LegalAgent

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="法律智能助手",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 自定义样式
# ============================================================
st.markdown("""
<style>
    /* 隐藏默认 footer */
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* 侧边栏背景 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #d0d0d0;
    }
    section[data-testid="stSidebar"] button {
        border-radius: 8px !important;
        font-size: 13px !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08);
    }

    /* 新建会话 — 紫色渐变 */
    section[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #818cf8) !important;
        border: none !important;
        color: #fff !important;
    }

    /* 会话列表按钮 — 深色半透明，融入背景 */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #c8c8d0 !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.12) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: #fff !important;
    }

    /* 聊天气泡 */
    div[data-testid="stChatMessage"] {
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin-bottom: 8px !important;
    }

    /* 回答气泡 — 浅蓝背景 */
    div[data-testid="stChatMessage"]:not(:has(button)) {
        background: linear-gradient(135deg, #f8faff, #f0f4ff);
    }

    /* 欢迎区域 */
    .welcome-box {
        text-align: center;
        padding: 32px 20px 8px;
    }

    /* 快捷问题按钮 */
    .st-key-quick_0 button, .st-key-quick_1 button,
    .st-key-quick_2 button, .st-key-quick_3 button {
        height: 64px !important;
        text-align: left !important;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        background: #ffffff !important;
        color: #1e293b !important;
        font-size: 13px !important;
        transition: all 0.15s !important;
    }
    .st-key-quick_0 button:hover, .st-key-quick_1 button:hover,
    .st-key-quick_2 button:hover, .st-key-quick_3 button:hover {
        border-color: #818cf8 !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.12) !important;
    }

    /* 输入框 */
    textarea[data-testid="stChatInput"] {
        border-radius: 12px !important;
        font-size: 15px !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 状态初始化
# ============================================================
if "agent" not in st.session_state:
    with st.spinner("系统初始化中..."):
        st.session_state.agent = LegalAgent()

if "sessions" not in st.session_state:
    initial_id = str(uuid.uuid4())
    st.session_state.sessions = {initial_id: {"title": "新会话", "messages": [], "history": []}}
    st.session_state.current_session_id = initial_id


# ============================================================
# 辅助函数
# ============================================================
def create_new_session():
    new_id = str(uuid.uuid4())
    st.session_state.sessions[new_id] = {"title": "新会话", "messages": [], "history": []}
    st.session_state.current_session_id = new_id

def switch_session(sid):
    st.session_state.current_session_id = sid
    st.session_state.agent.chat_history = st.session_state.sessions[sid]["history"]

def delete_session(sid):
    if len(st.session_state.sessions) > 1:
        del st.session_state.sessions[sid]
        if st.session_state.current_session_id == sid:
            st.session_state.current_session_id = next(iter(st.session_state.sessions))
            switch_session(st.session_state.current_session_id)
    else:
        create_new_session()
        del st.session_state.sessions[sid]


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="text-align:center; padding:8px 0 2px 0;">
        <div style="font-size:38px;">⚖️</div>
        <div style="font-size:18px; font-weight:700;">法律智能助手</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # 新建会话
    if st.button("＋ 新建会话", use_container_width=True, type="primary"):
        create_new_session()
        st.rerun()

    st.divider()

    # 会话列表 — 每个会话用 container 美化
    st.caption("💬 历史会话")

    session_ids = list(st.session_state.sessions.keys())
    for sid in session_ids:
        is_active = (sid == st.session_state.current_session_id)
        title = st.session_state.sessions[sid]["title"][:14]

        # 选中行高亮背景
        if is_active:
            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.25); border-radius:8px;
                        padding:6px 10px; margin:2px 0; border-left:3px solid #818cf8;">
                <span style="font-size:13px; font-weight:600;">● {title}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button(f"○  {title}", key=f"sess_{sid}", use_container_width=True):
                switch_session(sid)
                st.rerun()

    # 删除按钮
    st.divider()
    if len(session_ids) > 1:
        if st.button("🗑 删除当前会话", use_container_width=True, type="secondary"):
            delete_session(st.session_state.current_session_id)
            st.rerun()

    # 底部免责
    st.caption("⚠️ AI 回答仅供参考，不构成法律建议")


# ============================================================
# 主界面
# ============================================================
current_session = st.session_state.sessions[st.session_state.current_session_id]

# 欢迎页
if not current_session["messages"]:
    st.markdown("""
    <div class="welcome-box">
        <div style="font-size:56px; margin-bottom:12px;">⚖️</div>
        <div style="font-size:26px; font-weight:700; color:#1e293b; margin-bottom:6px;">您好，我是法律智能助手</div>
        <div style="font-size:15px; color:#64748b; margin-bottom:28px;">
            基于 RAG 检索增强生成 · 法条检索 · 案例分析 · 赔偿计算
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 快捷问题
    st.markdown("<div style='text-align:center; color:#94a3b8; font-size:12px; margin-bottom:10px;'>试试问我...</div>",
                unsafe_allow_html=True)

    quick_questions = [
        ("💰 工资拖欠", "公司拖欠了我3个月工资，应该怎么维权？", "0"),
        ("📋 劳动合同", "公司不和我签劳动合同，有什么赔偿？", "1"),
        ("🏠 租房纠纷", "房东不退押金，我该怎么处理？", "2"),
        ("🚗 交通事故", "被车撞了对方全责，能要多少赔偿？", "3"),
    ]

    cols = st.columns(2)
    for i, (label, desc, _) in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(label, help=desc, key=f"quick_{i}", use_container_width=True):
                st.session_state["quick_input"] = desc
                st.rerun()

# 对话历史
for message in current_session["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 输入处理
quick_input = st.session_state.pop("quick_input", None)
prompt = quick_input or st.chat_input("请描述您的法律问题...")

if prompt:
    # 更新标题
    if not current_session["messages"]:
        current_session["title"] = prompt[:16]

    # 用户消息
    current_session["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 助手回答
    with st.chat_message("assistant"):
        with st.status("🔍 正在分析您的问题...", expanded=False) as status:
            log_placeholder = st.empty()
            logs = []

            def ui_logger(msg):
                logs.append(msg)
                log_placeholder.code("\n".join(logs[-3:]), language="text")

            st.session_state.agent.logger = ui_logger
            st.session_state.agent.chat_history = current_session["history"]
            status.update(label="📖 正在生成专业回答...", state="running")

        full_response = ""
        message_placeholder = st.empty()

        for chunk in st.session_state.agent.answer_question_stream(prompt):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        current_session["messages"].append({"role": "assistant", "content": full_response})
        current_session["history"] = st.session_state.agent.chat_history

        status.update(label="✅ 分析完成", state="complete")
        st.rerun()
