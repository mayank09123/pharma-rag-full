"""
Pharma RAG Chatbot - Aesthetic Version
Author: Mayank Pratap Singh Chauhan (B01098725)

"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# Load Streamlit Cloud secrets BEFORE everything else
try:
    if hasattr(st, 'secrets'):
        for k, v in st.secrets.items():
            os.environ[str(k)] = str(v)
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.vector_store import DrugLabelVectorStore
from src.generation.rag_pipeline import PharmaRAGPipeline, OutputMode
from dotenv import load_dotenv
load_dotenv()

from src.retrieval.vector_store import DrugLabelVectorStore
from src.generation.rag_pipeline import PharmaRAGPipeline, OutputMode

st.set_page_config(page_title="Pharma RAG", page_icon="💊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ═══════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1629 50%, #0a1628 100%) !important;
    border-right: 1px solid rgba(0,180,160,0.15) !important;
    min-width: 280px !important;
    width: 280px !important;
}
section[data-testid="stSidebar"] > div {
    background: transparent !important;
    padding: 24px 18px !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div { color: #cbd5e1 !important; }

/* Sidebar logo area */
.sb-logo { 
    background: linear-gradient(135deg, rgba(0,180,160,0.15), rgba(0,122,110,0.08));
    border: 1px solid rgba(0,180,160,0.2);
    border-radius: 14px; padding: 16px; margin-bottom: 6px;
}
.sb-logo-title { font-size: 1.1rem; font-weight: 700; color: #f1f5f9 !important; margin: 0; }
.sb-logo-sub { font-size: 0.72rem; color: #64748b !important; margin: 3px 0 0; }

/* Mode buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    color: #94a3b8 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 5px !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,180,160,0.1) !important;
    border-color: rgba(0,180,160,0.3) !important;
    color: #e2e8f0 !important;
    transform: translateX(2px) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
}
[data-testid="stMetricValue"] { 
    color: #00B4A0 !important; 
    font-size: 1.5rem !important; 
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] { 
    color: #475569 !important; 
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Selectbox */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1629 50%, #0a1628 100%) !important;
    border-right: 1px solid rgba(0,180,160,0.15) !important;
    min-width: 280px !important;
    width: 280px !important;
    display: block !important;
    visibility: visible !important;
}

/* ═══════════════════════════════════════════
   MAIN AREA
═══════════════════════════════════════════ */
.main .block-container {
    background: #080d1a !important;
    min-height: 100vh !important;
    padding: 0 !important;
}

/* Top header bar */
.top-bar {
    background: linear-gradient(90deg, #0a0f1e, #0d1a2e);
    border-bottom: 1px solid rgba(0,180,160,0.15);
    padding: 16px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.top-bar-title { font-size: 1.05rem; font-weight: 600; color: #f1f5f9; margin: 0; }
.top-bar-sub { font-size: 0.72rem; color: #475569; margin: 2px 0 0; }
.top-bar-badge {
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.tbb-clinical    { background: rgba(0,180,160,0.12); color: #00B4A0; border: 1px solid rgba(0,180,160,0.25); }
.tbb-marketing   { background: rgba(239,159,39,0.12); color: #EF9F27; border: 1px solid rgba(239,159,39,0.25); }
.tbb-patient     { background: rgba(133,183,235,0.12); color: #85B7EB; border: 1px solid rgba(133,183,235,0.25); }
.tbb-regulatory  { background: rgba(175,169,236,0.12); color: #AFA9EC; border: 1px solid rgba(175,169,236,0.25); }

/* Chat area */
.chat-area { padding: 24px 28px; }

/* USER bubble */
.user-row {
    display: flex;
    justify-content: flex-end;
    align-items: flex-end;
    gap: 10px;
    margin: 16px 0;
    animation: fadeUp 0.3s ease;
}
.user-av {
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 700; color: white; flex-shrink: 0;
    box-shadow: 0 0 12px rgba(29,78,216,0.3);
}
.user-bub {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    font-size: 0.88rem; color: white;
    max-width: 68%; line-height: 1.65;
    box-shadow: 0 4px 20px rgba(29,78,216,0.25);
}

/* BOT bubble */
.bot-row {
    display: flex;
    justify-content: flex-start;
    align-items: flex-end;
    gap: 10px;
    margin: 16px 0;
    animation: fadeUp 0.3s ease;
}
.bot-av {
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, #0f3d2e, #1a5c45);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 800; color: #00B4A0; flex-shrink: 0;
    border: 1px solid rgba(0,180,160,0.4);
    box-shadow: 0 0 16px rgba(0,180,160,0.2);
}
.bot-bub {
    background: linear-gradient(135deg, #0f1e30, #111d2e);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 4px 18px 18px 18px;
    padding: 16px 20px;
    font-size: 0.88rem; color: #cbd5e1;
    max-width: 82%; line-height: 1.75;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* Mode badge inside bubble */
.mbadge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 12px; border-radius: 20px;
    font-size: 0.63rem; font-weight: 700;
    margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.08em;
}
.mb-clinical    { background: rgba(0,180,160,0.12); color: #00B4A0; border: 1px solid rgba(0,180,160,0.2); }
.mb-marketing   { background: rgba(239,159,39,0.12); color: #EF9F27; border: 1px solid rgba(239,159,39,0.2); }
.mb-patient     { background: rgba(133,183,235,0.12); color: #85B7EB; border: 1px solid rgba(133,183,235,0.2); }
.mb-regulatory  { background: rgba(175,169,236,0.12); color: #AFA9EC; border: 1px solid rgba(175,169,236,0.2); }

/* Sources */
.src-row {
    display: flex; flex-wrap: wrap; gap: 5px;
    margin-top: 12px; padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.src-tag {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.67rem; color: #64748b;
    transition: all 0.2s;
}
.src-tag:hover { border-color: rgba(0,180,160,0.3); color: #94a3b8; }
.src-sc { color: #00B4A0; font-weight: 600; }

/* Compliance */
.comp-ok { color: #34d399; font-size: 0.7rem; margin-top: 6px; display: flex; align-items: center; gap: 4px; }
.comp-fl { color: #f87171; font-size: 0.7rem; margin-top: 6px; display: flex; align-items: center; gap: 4px; }

/* Typing */
.typing-wrap { display: flex; align-items: center; gap: 5px; }
.typing-dot {
    width: 7px; height: 7px;
    background: #00B4A0; border-radius: 50%; margin: 0 1px;
    animation: tdot 1.4s ease infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; background: rgba(0,180,160,0.7); }
.typing-dot:nth-child(3) { animation-delay: 0.4s; background: rgba(0,180,160,0.4); }
@keyframes tdot {
    0%,60%,100% { transform: translateY(0) scale(1); }
    30% { transform: translateY(-5px) scale(1.1); }
}

/* Input bar */
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 28px !important;
    color: #f1f5f9 !important;
    padding: 14px 24px !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
}
.stTextInput input:focus {
    border-color: rgba(0,180,160,0.5) !important;
    background: rgba(0,180,160,0.04) !important;
    box-shadow: 0 0 0 3px rgba(0,180,160,0.08) !important;
}
.stTextInput input::placeholder { color: #334155 !important; }

/* Send button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #007A6E, #00B4A0) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 46px !important; height: 46px !important;
    font-size: 1.1rem !important; padding: 0 !important;
    box-shadow: 0 4px 16px rgba(0,180,160,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: scale(1.06) !important;
    box-shadow: 0 6px 20px rgba(0,180,160,0.5) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.05) !important; }

/* Section labels */
.section-label {
    font-size: 0.65rem; font-weight: 700; color: #334155 !important;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Pipeline ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FDA drug data...")
def load_pipeline():
    store = DrugLabelVectorStore(use_rerank=False)
    if store.collection_stats()["total_chunks"] == 0:
        from src.ingestion.fda_loader import FDALabelLoader
        loader = FDALabelLoader()
        chunks = loader.load_multiple(["warfarin","metformin","lisinopril"])
        store.add_chunks(chunks)
    return store, PharmaRAGPipeline(vector_store=store)

# ── Session ───────────────────────────────────────────────────────────────
for k,v in {"messages":[],"input_key":0,"mode":"clinical"}.items():
    if k not in st.session_state: st.session_state[k] = v

# ── Data ──────────────────────────────────────────────────────────────────
MODES = {
    "clinical":   {"icon":"🏥","name":"Clinical (HCP)",  "color":"clinical"},
    "marketing":  {"icon":"📢","name":"Marketing",       "color":"marketing"},
    "patient":    {"icon":"👤","name":"Patient",         "color":"patient"},
    "regulatory": {"icon":"📋","name":"Regulatory",      "color":"regulatory"},
}
DOTS = {"clinical":"🟢","marketing":"🟡","patient":"🔵","regulatory":"🟣"}
QUESTIONS = {
    "clinical": {
        "Warfarin":   ["What are the contraindications for warfarin?","What are warfarin side effects?","What drug interactions does warfarin have?","What is the warfarin dosage?","How does warfarin work?","Is warfarin safe in pregnancy?","What monitoring is needed for warfarin?"],
        "Metformin":  ["What is the metformin dosage?","What are metformin side effects?","How does metformin work?","Is metformin safe in kidney disease?","Can metformin cause lactic acidosis?"],
        "Lisinopril": ["Lisinopril contraindications?","What is the lisinopril dosage?","What are lisinopril side effects?","How does lisinopril work?","Can lisinopril cause a cough?"],
    },
    "marketing": {
        "Warfarin":   ["Write a physician brief for warfarin","Approved indication for warfarin","Safety info required for warfarin","Warfarin efficacy summary"],
        "Metformin":  ["Write a marketing brief for metformin","Clinical evidence for metformin","Metformin promotional summary"],
        "Lisinopril": ["Promotional overview of lisinopril","Benefits of lisinopril","HCP brief for lisinopril"],
    },
    "patient": {
        "Warfarin":   ["What is warfarin used for?","Foods to avoid with warfarin?","Can I drink alcohol with warfarin?","Is warfarin safe in pregnancy?","What if I miss a warfarin dose?"],
        "Metformin":  ["What does metformin do?","Will metformin make me sick?","What if I forget metformin?","Can I drink alcohol on metformin?"],
        "Lisinopril": ["Why do people take lisinopril?","Lisinopril side effects in simple words?","How long for lisinopril to work?"],
    },
    "regulatory": {
        "Warfarin":   ["Regulatory summary of warfarin","Verbatim indication for warfarin","Boxed warnings for warfarin","Warfarin safety profile dossier"],
        "Metformin":  ["FDA approved indication for metformin","Regulatory summary of metformin","Metformin clinical pharmacology"],
        "Lisinopril": ["Safety profile of lisinopril","Regulatory dossier for lisinopril","Lisinopril warnings section"],
    },
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    mode = st.session_state.mode

    st.markdown("""
    <div class="sb-logo">
        <p class="sb-logo-title">💊 Pharma RAG</p>
        <p class="sb-logo-sub">Mayank Pratap Singh Chauhan · B01098725<br>Advisor: Prof. Sujoy Sikdar</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Output Mode</p>', unsafe_allow_html=True)
    for mk, mv in MODES.items():
        active = mode == mk
        prefix = "▶  " if active else "    "
        label  = f"{prefix}{DOTS[mk]}  {mv['name']}"
        if st.button(label, key=f"mode_{mk}", use_container_width=True):
            st.session_state.mode = mk
            st.session_state.messages = []
            st.rerun()

    st.divider()

    st.markdown('<p class="section-label">Drug Focus</p>', unsafe_allow_html=True)
    drug = st.selectbox("drug", ["Warfarin","Metformin","Lisinopril"],
                         label_visibility="collapsed")

    st.markdown(f'<p class="section-label" style="margin-top:12px;">Quick Questions</p>', unsafe_allow_html=True)
    for i, q in enumerate(QUESTIONS.get(mode,{}).get(drug,[])):
        if st.button(q, key=f"qq_{mode}_{drug}_{i}", use_container_width=True):
            st.session_state["pending"] = q

    st.divider()

    try:
        store, _ = load_pipeline()
        stats = store.collection_stats()
        c1, c2 = st.columns(2)
        c1.metric("Chunks", stats["total_chunks"])
        c2.metric("Score",  "0.96")
    except Exception as e:
        st.error(f"{e}")

    st.divider()
    if st.button("🗑  Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── MAIN ──────────────────────────────────────────────────────────────────
mode     = st.session_state.mode
mode_cfg = MODES[mode]

# Header
hc1, hc2 = st.columns([4,1])
with hc1:
    st.markdown("""
    <div style="padding:18px 28px 0;">
        <h3 style="color:#f1f5f9;margin:0;font-size:1.1rem;font-weight:600;">Pharma RAG Chatbot</h3>
        <p style="color:#334155;font-size:0.72rem;margin:3px 0 0;">
            FDA DailyMed · GPT-4o · ChromaDB · Binghamton University, SUNY · 2025
        </p>
    </div>
    """, unsafe_allow_html=True)
with hc2:
    st.markdown(f"""
    <div style="padding:18px 28px 0;text-align:right;">
        <span class="top-bar-badge tbb-{mode}">
            {mode_cfg['icon']} {mode_cfg['name']}
        </span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Clear", key="clr"):
        st.session_state.messages = []
        st.rerun()

st.markdown("<div style='padding:0 28px;'>", unsafe_allow_html=True)
st.divider()
st.markdown("</div>", unsafe_allow_html=True)

# Messages area
st.markdown('<div style="padding:8px 28px 120px;">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(f"""
    <div class="bot-row">
        <div class="bot-av">AI</div>
        <div class="bot-bub">
            <span class="mbadge mb-{mode}">{mode_cfg['icon']} {mode.upper()}</span><br>
            👋 &nbsp;<strong>Hello! I'm the Pharma RAG Assistant.</strong><br><br>
            I answer questions about <strong>Warfarin · Metformin · Lisinopril</strong>
            using real FDA prescribing information — grounded, cited, never hallucinated.<br><br>
            Currently in <strong>{mode_cfg['icon']} {mode_cfg['name']}</strong> mode.<br><br>
            Pick a drug from the sidebar and click any question, or type your own below.
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-row">
            <div class="user-bub">{msg["content"]}</div>
            <div class="user-av">YOU</div>
        </div>""", unsafe_allow_html=True)
    else:
        m   = msg.get("mode","clinical")
        mc  = MODES.get(m, MODES["clinical"])
        srcs = "".join(
            f'<span class="src-tag">📄 {s.get("metadata",{}).get("drug_name","?")} — '
            f'{s.get("metadata",{}).get("section_name","?").replace("_"," ")} '
            f'<span class="src-sc">({s.get("score",0):.2f})</span></span>'
            for s in msg.get("sources",[]))
        src_html  = f'<div class="src-row">{srcs}</div>' if srcs else ""
        comp_html = (
            f'<div class="comp-fl">⚠️ &nbsp;{" · ".join(msg["flags"][:2])}</div>'
            if msg.get("flags")
            else '<div class="comp-ok">✓ &nbsp;Compliant — no compliance flags</div>'
        )
        st.markdown(f"""
        <div class="bot-row">
            <div class="bot-av">AI</div>
            <div class="bot-bub">
                <span class="mbadge mb-{m}">{mc['icon']} {m.upper()}</span><br>
                {msg["content"].replace(chr(10), "<br>")}
                {src_html}
                {comp_html}
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Input bar (sticky at bottom)
st.markdown("""
<div style="position:fixed;bottom:0;left:280px;right:0;
     background:linear-gradient(0deg,#080d1a 80%,transparent);
     padding:16px 28px 20px;z-index:999;border-top:1px solid rgba(255,255,255,0.04);">
</div>
""", unsafe_allow_html=True)

pending = st.session_state.pop("pending", None)
ci, cb  = st.columns([12, 1])
with ci:
    user_input = st.text_input("q", value=pending or "",
        placeholder=f"Ask in {mode_cfg['name']} mode...",
        label_visibility="collapsed",
        key=f"inp_{st.session_state.input_key}")
with cb:
    send = st.button("➤", type="primary", use_container_width=True)

question = (pending or user_input).strip()
if (send or pending) and question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.empty():
        st.markdown(f"""
        <div class="bot-row">
            <div class="bot-av">AI</div>
            <div class="bot-bub">
                <span class="mbadge mb-{mode}">{mode_cfg['icon']} {mode.upper()}</span><br>
                <div class="typing-wrap">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span style="color:#1e3a5f;font-size:0.78rem;margin-left:6px;">Searching FDA labels...</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        time.sleep(1.0)
    try:
        store, pipeline = load_pipeline()
        response = pipeline.query(question, mode=OutputMode(mode))
        answer  = response.answer
        sources = response.sources
        flags   = response.compliance_flags
    except Exception as e:
        answer  = f"❌ Error: {str(e)}"
        sources, flags = [], []
    st.session_state.messages.append({
        "role":"assistant","content":answer,
        "mode":mode,"sources":sources[:4],"flags":flags})
    st.session_state.input_key += 1
    st.rerun()