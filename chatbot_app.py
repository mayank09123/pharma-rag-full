"""
Pharma RAG Chatbot

Author: Mayank Pratap Singh Chauhan 


Run: python -m streamlit run chatbot_app.py
"""
import streamlit_startup
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os
from dotenv import load_dotenv

load_dotenv()

st.write("DEBUG OPENAI KEY:", os.getenv("OPENAI_API_KEY"))
import streamlit as st
from src.retrieval.vector_store import DrugLabelVectorStore
from src.generation.rag_pipeline import PharmaRAGPipeline, OutputMode

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma RAG Chatbot",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Hide default streamlit header */
#MainMenu, footer { visibility: hidden; }
section[data-testid="stSidebar"] {
    min-width: 320px !important;
    max-width: 320px !important;
}
/* Chat container */
.chat-header {
    background: linear-gradient(135deg, #0B1F3A 0%, #007A6E 100%);
    padding: 20px 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    color: white;
}
.chat-header h1 { margin:0; font-size:1.5rem; font-weight:700; }
.chat-header p  { margin:4px 0 0; font-size:0.8rem; opacity:0.8; }

/* Bot message bubble */
.bot-bubble {
    background: #F0FDF9;
    border: 1px solid #D4F1EE;
    border-left: 4px solid #007A6E;
    border-radius: 0 12px 12px 12px;
    padding: 14px 18px;
    margin: 8px 0 8px 8px;
    max-width: 85%;
    font-size: 0.92rem;
    line-height: 1.65;
    color: #1C2B3A;
}

/* User message bubble */
.user-bubble {
    background: linear-gradient(135deg, #0B1F3A, #007A6E);
    border-radius: 12px 0 12px 12px;
    padding: 12px 18px;
    margin: 8px 8px 8px auto;
    max-width: 75%;
    font-size: 0.92rem;
    color: white;
    text-align: right;
    display: block;
}

/* Source badge */
.source-badge {
    display: inline-block;
    background: #D4F1EE;
    color: #007A6E;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    margin: 2px 2px 6px;
    border: 1px solid #007A6E;
}

/* Flag badge */
.flag-badge {
    display: inline-block;
    background: #FEE2E2;
    color: #B91C1C;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    margin: 2px;
    border: 1px solid #B91C1C;
}

/* Clean badge */
.clean-badge {
    display: inline-block;
    background: #D1FAE5;
    color: #059669;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    margin: 2px;
    border: 1px solid #059669;
}

/* Mode badge */
.mode-badge {
    display: inline-block;
    background: #0B1F3A;
    color: #00B4A0;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 10px;
    margin-bottom: 8px;
    letter-spacing: 0.05em;
}

/* Sidebar */
.sidebar-card {
    background: #F4F6F8;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    color: #1C2B3A;
    cursor: pointer;
    border: 1px solid #E2E8F0;
    transition: all 0.2s;
}

/* Typing dots */
.typing-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #007A6E;
    border-radius: 50%;
    margin: 0 2px;
    animation: bounce 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%,60%,100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
}

.stTextInput input {
    border-radius: 24px !important;
    border: 2px solid #D4F1EE !important;
    padding: 10px 18px !important;
    font-size: 0.92rem !important;
}
.stTextInput input:focus {
    border-color: #007A6E !important;
    box-shadow: 0 0 0 2px rgba(0,180,160,0.15) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load pipeline ─────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    store    = DrugLabelVectorStore()
    pipeline = PharmaRAGPipeline(vector_store=store)
    return store, pipeline

# ── Session state ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💊 Pharma RAG Chat")

    st.caption("Mayank Pratap Singh Chauhan ")
    st.divider()

    # Mode switcher
    mode_label = st.selectbox(
        "🎯 Output Mode",
        ["clinical", "marketing", "patient", "regulatory"],
        format_func=lambda x: {
            "clinical":   "🏥 Clinical (HCP)",
            "marketing":  "📢 Marketing",
            "patient":    "👤 Patient (Plain)",
            "regulatory": "📋 Regulatory",
        }[x],
    )

    st.divider()

    # Drug selector
    drug_focus = st.selectbox(
        "💊 Drug Focus",
        ["All drugs", "Warfarin", "Metformin", "Lisinopril"],
    )

    st.divider()

    # Quick questions
    st.markdown("**💡 Quick Questions**")
    quick_qs = [
        "What are the contraindications for warfarin?",
        "What is the dosage for metformin?",
        "Side effects of lisinopril?",
        "Drug interactions with warfarin?",
        "Is warfarin safe in pregnancy?",
        "How does metformin work?",
        "Warfarin monitoring requirements?",
        "Lisinopril for heart failure?",
    ]
    for q in quick_qs:
        if st.button(q, use_container_width=True, key=f"qq_{q[:20]}"):
            st.session_state["pending_question"] = q

    st.divider()

    # Vector store stats
    try:
        store, _ = load_pipeline()
        stats = store.collection_stats()
        col1, col2 = st.columns(2)
        col1.metric("Chunks", stats["total_chunks"])
        col2.metric("Status", "✅" if stats["total_chunks"] > 0 else "❌")
    except:
        st.error("Vector store not ready")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
  <h1>💊 Pharma RAG Chatbot</h1>
  <p>Grounded answers from FDA-approved drug Label 2025</p>
</div>
""", unsafe_allow_html=True)

# ── Welcome message ───────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="bot-bubble">
        👋 <b>Hello! I'm the Pharma RAG Assistant.</b><br><br>
        I answer questions about FDA-approved drugs using real prescribing information from DailyMed.
        Every answer is grounded in official labeling data — no hallucinations.<br><br>
        <b>Currently loaded drugs:</b> Warfarin · Metformin · Lisinopril<br><br>
        Try asking: <i>"What are the contraindications for warfarin?"</i>
    </div>
    """, unsafe_allow_html=True)

# ── Render chat history ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        mode_icon = {"clinical":"🏥","marketing":"📢","patient":"👤","regulatory":"📋"}.get(msg.get("mode","clinical"),"💊")
        mode_name = msg.get("mode","clinical").upper()

        sources_html = ""
        for s in msg.get("sources", []):
            drug    = s.get("metadata", {}).get("drug_name", "")
            section = s.get("metadata", {}).get("section_name", "")
            score   = s.get("score", 0)
            sources_html += f'<span class="source-badge">📄 {drug} — {section} ({score:.2f})</span>'

        flags_html = ""
        for flag in msg.get("flags", []):
            flags_html += f'<span class="flag-badge">⚠️ {flag[:50]}</span>'
        if not msg.get("flags"):
            flags_html = '<span class="clean-badge">✅ Compliant</span>'

        st.markdown(f"""
        <div class="bot-bubble">
            <span class="mode-badge">{mode_icon} {mode_name}</span><br>
            {msg["content"].replace(chr(10), "<br>")}
            <br><br>
            <div style="border-top:1px solid #D4F1EE;padding-top:8px;margin-top:4px;">
                <div style="margin-bottom:4px;">{sources_html}</div>
                {flags_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Handle pending question from sidebar ─────────────────────────────────
pending = st.session_state.pop("pending_question", None)

# ── Chat input ────────────────────────────────────────────────────────────
col_input, col_send = st.columns([9, 1])
with col_input:
    user_input = st.text_input(
        "message",
        value=pending or "",
        placeholder="Ask about any drug... e.g. What are warfarin contraindications?",
        label_visibility="collapsed",
        key=f"chat_input_{st.session_state.input_key}",
    )
with col_send:
    send = st.button("➤", type="primary", use_container_width=True)

# ── Process input ─────────────────────────────────────────────────────────
question = (pending or user_input).strip()

if (send or pending) and question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})

    # Typing animation placeholder
    with st.empty():
        st.markdown("""
        <div class="bot-bubble">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.6)

    # Run RAG pipeline
    try:
        store, pipeline = load_pipeline()

        if store.collection_stats()["total_chunks"] == 0:
            answer  = "⚠️ Vector store is empty. Please run `python scripts/ingest.py` first."
            sources = []
            flags   = []
        else:
            drug_filter = None if drug_focus == "All drugs" else drug_focus.lower()
            response = pipeline.query(
                question,
                mode=OutputMode(mode_label),
                drug_filter=drug_filter,
            )
            answer  = response.answer
            sources = response.sources
            flags   = response.compliance_flags

    except Exception as e:
        answer  = f"❌ Error: {str(e)}"
        sources = []
        flags   = []

    # Save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "mode":    mode_label,
        "sources": sources[:4],
        "flags":   flags,
    })

    # Clear input
    st.session_state.input_key += 1
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown(
    "<br><center><small style='color:#9CA3AF'>Built on FDA DailyMed · GPT-4o · ChromaDB · "
"· 2025</small></center>",

    unsafe_allow_html=True,
)
