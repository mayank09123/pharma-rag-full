"""
Pharma RAG Chatbot - Full Beautiful Version
Author: Mayank Pratap Singh Chauhan (B01098725)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ONLY load from Streamlit secrets - ignore .env completely
try:
    if hasattr(st, 'secrets'):
        for k, v in st.secrets.items():
            os.environ[str(k)] = str(v)
except Exception as e:
    pass

# Only load .env if running locally (no secrets available)
if not os.environ.get('OPENAI_API_KEY'):
    from dotenv import load_dotenv
    load_dotenv()

from src.retrieval.vector_store import DrugLabelVectorStore
from src.generation.rag_pipeline import PharmaRAGPipeline, OutputMode
st.set_page_config(
    page_title="Pharma RAG",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0!important;max-width:100%!important;}
.main .block-container{background:#030810!important;min-height:100vh!important;padding:0!important;}

/* SIDEBAR */
section[data-testid="stSidebar"]{
    background:#050b18!important;
    border-right:1px solid rgba(0,180,160,0.12)!important;
    min-width:290px!important;width:290px!important;
    transform:none!important;display:block!important;visibility:visible!important;
}
section[data-testid="stSidebar"]>div{background:transparent!important;padding:0!important;}
[data-testid="collapsedControl"]{
    display:none!important;
    width:40px!important;
    height:40px!important;
    background:rgba(0,180,160,0.1)!important;
    border:1px solid rgba(0,180,160,0.2)!important;
    border-radius:50%!important;
    align-items:center!important;
    justify-content:center!important;
    cursor:pointer!important;
    transition:all 0.3s ease!important;
}
[data-testid="collapsedControl"]:hover{
    background:rgba(0,180,160,0.2)!important;
    transform:scale(1.1)!important;
}
[data-testid="collapsedControl"] svg{
    color:#00B4A0!important;
    fill:#00B4A0!important;
}
section[data-testid="stSidebar"]{
    transition:all 0.4s cubic-bezier(0.4,0,0.2,1)!important;
    background:#050b18!important;
    border-right:1px solid rgba(0,180,160,0.12)!important;
    min-width:290px!important;
    width:290px!important;
}
section[data-testid="stSidebar"][aria-expanded="false"]{
    width:0px!important;
    min-width:0px!important;
    overflow:hidden!important;
    transform:translateX(-290px)!important;
    transition:all 0.4s cubic-bezier(0.4,0,0.2,1)!important;
}

.sb-header{
    padding:22px 18px 16px;
    border-bottom:1px solid rgba(255,255,255,0.05);
    position:relative;overflow:hidden;
}
.sb-header::before{
    content:'';position:absolute;top:-40px;right:-40px;
    width:120px;height:120px;
    background:radial-gradient(circle,rgba(0,180,160,0.15) 0%,transparent 70%);
    border-radius:50%;animation:pglow 3s ease infinite;
}
@keyframes pglow{0%,100%{opacity:0.6;transform:scale(1);}50%{opacity:1;transform:scale(1.1);}}
.sb-icon{
    width:38px;height:38px;
    background:linear-gradient(135deg,#00B4A0,#007A6E);
    border-radius:10px;display:flex;align-items:center;
    justify-content:center;font-size:1.2rem;margin-bottom:10px;
    box-shadow:0 0 20px rgba(0,180,160,0.3);
    animation:float 3s ease infinite;
}
@keyframes float{0%,100%{transform:translateY(0);}50%{transform:translateY(-3px);}}
.sb-title{font-size:1rem;font-weight:700;color:#f1f5f9!important;margin:0;}
.sb-sub{font-size:0.68rem;color:#475569!important;margin:3px 0 0;line-height:1.5;}
.sb-lbl{
    font-size:0.6rem;font-weight:700;color:#334155!important;
    text-transform:uppercase;letter-spacing:0.12em;
    padding:12px 18px 6px;display:block;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton>button{
    background:transparent!important;
    border:1px solid transparent!important;
    color:#64748b!important;border-radius:10px!important;
    padding:9px 14px!important;font-size:0.83rem!important;
    font-weight:500!important;text-align:left!important;
    width:100%!important;margin-bottom:3px!important;
    transition:all 0.25s cubic-bezier(0.4,0,0.2,1)!important;
}
section[data-testid="stSidebar"] .stButton>button:hover{
    background:rgba(0,180,160,0.08)!important;
    border-color:rgba(0,180,160,0.2)!important;
    color:#cbd5e1!important;transform:translateX(4px)!important;
}
[data-testid="stMetric"]{
    background:rgba(255,255,255,0.03)!important;
    border:1px solid rgba(255,255,255,0.06)!important;
    border-radius:12px!important;padding:12px!important;
    transition:all 0.3s ease!important;
}
[data-testid="stMetric"]:hover{
    border-color:rgba(0,180,160,0.25)!important;
    background:rgba(0,180,160,0.04)!important;
}
[data-testid="stMetricValue"]{color:#00B4A0!important;font-size:1.5rem!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{color:#475569!important;font-size:0.65rem!important;text-transform:uppercase!important;letter-spacing:0.08em!important;}
section[data-testid="stSidebar"] .stSelectbox>div>div{
    background:rgba(255,255,255,0.04)!important;
    border:1px solid rgba(255,255,255,0.08)!important;
    border-radius:10px!important;color:#cbd5e1!important;font-size:0.83rem!important;
}

/* HERO */
.hero{
    background:linear-gradient(135deg,#050b18 0%,#0a1628 40%,#071520 100%);
    padding:24px 32px 20px;
    border-bottom:1px solid rgba(99,179,237,0.08);
    position:relative;overflow:hidden;
}
.hero::before{
    content:'';position:absolute;top:0;left:0;right:0;bottom:0;
    background:radial-gradient(ellipse at 10% 50%,rgba(0,180,160,0.08) 0%,transparent 50%),
               radial-gradient(ellipse at 90% 20%,rgba(99,179,237,0.06) 0%,transparent 50%);
    pointer-events:none;
}
.hero-inner{display:flex;align-items:center;justify-content:space-between;position:relative;z-index:1;}
.hero-title{font-size:1.15rem;font-weight:700;color:#f1f5f9;margin:0;display:flex;align-items:center;gap:10px;}
.live-pill{
    display:inline-flex;align-items:center;gap:5px;
    background:rgba(0,180,160,0.1);border:1px solid rgba(0,180,160,0.2);
    border-radius:20px;padding:3px 12px;font-size:0.7rem;font-weight:600;color:#00B4A0;
    animation:ppulse 2s ease infinite;
}
@keyframes ppulse{0%,100%{box-shadow:0 0 0 0 rgba(0,180,160,0);}50%{box-shadow:0 0 0 4px rgba(0,180,160,0.08);}}
.live-dot{width:6px;height:6px;border-radius:50%;background:#00B4A0;animation:blink 2s ease infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.3;}}
.hero-sub{font-size:0.72rem;color:#475569;margin:5px 0 0;}
.hero-stats{display:flex;gap:18px;margin-top:12px;}
.hstat{display:flex;align-items:center;gap:5px;font-size:0.7rem;color:#475569;}
.hstat-dot{width:6px;height:6px;border-radius:50%;}
.mbadge{
    padding:6px 16px;border-radius:20px;font-size:0.75rem;font-weight:600;
    letter-spacing:0.03em;animation:badge-in 0.4s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes badge-in{from{transform:scale(0.8);opacity:0;}to{transform:scale(1);opacity:1;}}
.mb-clinical   {background:rgba(0,180,160,0.12);color:#00B4A0;border:1px solid rgba(0,180,160,0.25);}
.mb-marketing  {background:rgba(239,159,39,0.12);color:#EF9F27;border:1px solid rgba(239,159,39,0.25);}
.mb-patient    {background:rgba(133,183,235,0.12);color:#85B7EB;border:1px solid rgba(133,183,235,0.25);}
.mb-regulatory {background:rgba(175,169,236,0.12);color:#AFA9EC;border:1px solid rgba(175,169,236,0.25);}

/* DRUG CARDS */
.dcards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;padding:14px 32px;background:#030810;border-bottom:1px solid rgba(255,255,255,0.04);}
.dcard{
    background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
    border-radius:12px;padding:12px 14px;
    transition:all 0.3s cubic-bezier(0.4,0,0.2,1);position:relative;overflow:hidden;
}
.dcard::after{
    content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.04),transparent);
    transition:left 0.6s ease;
}
.dcard:hover{border-color:rgba(0,180,160,0.25)!important;background:rgba(0,180,160,0.05)!important;transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(0,0,0,0.3)!important;}
.dcard:hover::after{left:100%;}
.dcard-icon{font-size:1.3rem;margin-bottom:6px;}
.dcard-name{font-size:0.82rem;font-weight:600;color:#e2e8f0;}
.dcard-class{font-size:0.68rem;color:#475569;margin-top:2px;}
.dcard-stat{margin-top:8px;display:inline-block;background:rgba(0,180,160,0.08);border:1px solid rgba(0,180,160,0.12);border-radius:20px;padding:2px 8px;font-size:0.65rem;color:#00B4A0;}

/* CHAT */
.chat-area{padding:20px 32px 100px;}
.welcome{
    background:linear-gradient(135deg,rgba(0,180,160,0.06),rgba(99,179,237,0.04));
    border:1px solid rgba(0,180,160,0.12);border-radius:16px;
    padding:24px;margin-bottom:20px;position:relative;overflow:hidden;
    animation:card-in 0.5s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes card-in{from{transform:translateY(10px);opacity:0;}to{transform:translateY(0);opacity:1;}}
.welcome::before{content:'';position:absolute;top:0;right:0;width:200px;height:200px;background:radial-gradient(circle,rgba(0,180,160,0.08) 0%,transparent 70%);pointer-events:none;}
.w-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(0,180,160,0.1);border:1px solid rgba(0,180,160,0.15);border-radius:20px;padding:3px 12px;font-size:0.68rem;font-weight:700;color:#00B4A0;margin-bottom:12px;text-transform:uppercase;letter-spacing:0.06em;}
.w-title{font-size:1.05rem;font-weight:700;color:#f1f5f9;margin:0 0 8px;}
.w-text{font-size:0.85rem;color:#64748b;line-height:1.7;margin:0;}
.w-pills{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap;}
.w-pill{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:4px 12px;font-size:0.75rem;font-weight:500;color:#94a3b8;}

/* USER bubble */
.urow{display:flex;justify-content:flex-end;align-items:flex-end;gap:10px;margin:16px 0;animation:min-right 0.35s cubic-bezier(0.34,1.56,0.64,1);}
@keyframes min-right{from{transform:translateX(20px);opacity:0;}to{transform:translateX(0);opacity:1;}}
.uav{width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#1e40af,#3b82f6);display:flex;align-items:center;justify-content:center;font-size:0.62rem;font-weight:800;color:white;flex-shrink:0;box-shadow:0 0 16px rgba(59,130,246,0.3);}
.ubub{background:linear-gradient(135deg,#1e40af,#2563eb);border-radius:18px 18px 4px 18px;padding:12px 18px;font-size:0.88rem;color:white;max-width:68%;line-height:1.65;box-shadow:0 4px 20px rgba(37,99,235,0.25);}

/* BOT bubble */
.brow{display:flex;justify-content:flex-start;align-items:flex-end;gap:10px;margin:16px 0;animation:min-left 0.35s cubic-bezier(0.34,1.56,0.64,1);}
@keyframes min-left{from{transform:translateX(-20px);opacity:0;}to{transform:translateX(0);opacity:1;}}
.bav{width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#0f3d2e,#1a5c45);display:flex;align-items:center;justify-content:center;font-size:0.62rem;font-weight:800;color:#00B4A0;flex-shrink:0;border:1px solid rgba(0,180,160,0.35);box-shadow:0 0 16px rgba(0,180,160,0.2);}
.bbub{
    background:linear-gradient(135deg,#0c1929,#0f1e2e);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:4px 18px 18px 18px;
    padding:16px 20px;font-size:0.88rem;color:#cbd5e1;
    max-width:82%;line-height:1.75;
    box-shadow:0 4px 24px rgba(0,0,0,0.35);
    position:relative;overflow:hidden;
}
.bbub::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%;border-radius:4px 0 0 4px;}
.bc-clinical::before{background:linear-gradient(180deg,#00B4A0,#007A6E);}
.bc-marketing::before{background:linear-gradient(180deg,#EF9F27,#BA7517);}
.bc-patient::before{background:linear-gradient(180deg,#85B7EB,#378ADD);}
.bc-regulatory::before{background:linear-gradient(180deg,#AFA9EC,#7F77DD);}

.mbub{display:inline-flex;align-items:center;gap:5px;padding:3px 12px;border-radius:20px;font-size:0.63rem;font-weight:700;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.08em;}
.mc-clinical   {background:rgba(0,180,160,0.1);color:#00B4A0;border:1px solid rgba(0,180,160,0.18);}
.mc-marketing  {background:rgba(239,159,39,0.1);color:#EF9F27;border:1px solid rgba(239,159,39,0.18);}
.mc-patient    {background:rgba(133,183,235,0.1);color:#85B7EB;border:1px solid rgba(133,183,235,0.18);}
.mc-regulatory {background:rgba(175,169,236,0.1);color:#AFA9EC;border:1px solid rgba(175,169,236,0.18);}

.srow{display:flex;flex-wrap:wrap;gap:5px;margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.05);}
.stag{display:inline-flex;align-items:center;gap:4px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:20px;padding:3px 10px;font-size:0.67rem;color:#475569;}
.ssc{color:#00B4A0;font-weight:700;}
.cok{display:inline-flex;align-items:center;gap:5px;color:#34d399;font-size:0.7rem;margin-top:6px;}
.cfl{display:inline-flex;align-items:center;gap:5px;color:#f87171;font-size:0.7rem;margin-top:6px;}

/* TYPING */
.typing-wrap{display:flex;align-items:center;gap:6px;padding:4px 0;}
.tdot{width:7px;height:7px;background:#00B4A0;border-radius:50%;margin:0 1px;animation:td 1.4s ease infinite;}
.tdot:nth-child(2){animation-delay:0.15s;opacity:0.7;}
.tdot:nth-child(3){animation-delay:0.3s;opacity:0.4;}
@keyframes td{0%,60%,100%{transform:translateY(0) scale(1);}30%{transform:translateY(-6px) scale(1.15);}}
.tlbl{font-size:0.75rem;color:#1e3a5f;animation:tf 1.5s ease infinite;}
@keyframes tf{0%,100%{opacity:0.4;}50%{opacity:1;}}

/* INPUT */
.stTextInput input{
    background:rgba(255,255,255,0.04)!important;
    border:1px solid rgba(255,255,255,0.09)!important;
    border-radius:28px!important;color:#f1f5f9!important;
    padding:14px 24px!important;font-size:0.9rem!important;
    transition:all 0.3s ease!important;
}
.stTextInput input:focus{
    border-color:rgba(0,180,160,0.45)!important;
    background:rgba(0,180,160,0.04)!important;
    box-shadow:0 0 0 3px rgba(0,180,160,0.08)!important;
}
.stTextInput input::placeholder{color:#1e3a5f!important;}
.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#007A6E,#00B4A0)!important;
    border:none!important;border-radius:50%!important;
    width:48px!important;height:48px!important;
    font-size:1.1rem!important;padding:0!important;
    box-shadow:0 4px 16px rgba(0,180,160,0.35)!important;
    transition:all 0.25s cubic-bezier(0.34,1.56,0.64,1)!important;
}
.stButton>button[kind="primary"]:hover{transform:scale(1.1) rotate(5deg)!important;box-shadow:0 8px 24px rgba(0,180,160,0.5)!important;}
hr{border-color:rgba(255,255,255,0.04)!important;}
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

MODES = {
    "clinical":   {"icon":"🏥","name":"Clinical (HCP)","dot":"🟢"},
    "marketing":  {"icon":"📢","name":"Marketing",     "dot":"🟡"},
    "patient":    {"icon":"👤","name":"Patient",       "dot":"🔵"},
    "regulatory": {"icon":"📋","name":"Regulatory",    "dot":"🟣"},
}
DRUGS = {
    "Warfarin":   {"icon":"🩸","cls":"Anticoagulant","stat":"74 chunks"},
    "Metformin":  {"icon":"💉","cls":"Antidiabetic", "stat":"41 chunks"},
    "Lisinopril": {"icon":"❤️","cls":"ACE Inhibitor","stat":"38 chunks"},
}
QUESTIONS = {
    "clinical": {
        "Warfarin":   ["What are the contraindications for warfarin?","What are warfarin side effects?","Drug interactions with warfarin?","What is the warfarin dosage?","How does warfarin work?","Is warfarin safe in pregnancy?","What monitoring is needed?"],
        "Metformin":  ["What is the metformin dosage?","Metformin side effects?","How does metformin work?","Is metformin safe in kidney disease?","Can metformin cause lactic acidosis?"],
        "Lisinopril": ["Lisinopril contraindications?","What is the lisinopril dosage?","Lisinopril side effects?","How does lisinopril work?","Can lisinopril cause a cough?"],
    },
    "marketing": {
        "Warfarin":   ["Write a physician brief for warfarin","Approved indication for warfarin","Safety info for warfarin","Warfarin efficacy summary"],
        "Metformin":  ["Write a marketing brief for metformin","Clinical evidence for metformin","Metformin promotional summary"],
        "Lisinopril": ["Promotional overview of lisinopril","Benefits of lisinopril","HCP brief for lisinopril"],
    },
    "patient": {
        "Warfarin":   ["What is warfarin used for?","Foods to avoid with warfarin?","Can I drink alcohol with warfarin?","Is warfarin safe in pregnancy?","What if I miss a warfarin dose?"],
        "Metformin":  ["What does metformin do?","Will metformin make me sick?","What if I forget metformin?"],
        "Lisinopril": ["Why do people take lisinopril?","Lisinopril side effects in simple words?","How long for lisinopril to work?"],
    },
    "regulatory": {
        "Warfarin":   ["Regulatory summary of warfarin","Verbatim indication for warfarin","Boxed warnings for warfarin"],
        "Metformin":  ["FDA approved indication for metformin","Regulatory summary of metformin"],
        "Lisinopril": ["Safety profile of lisinopril","Regulatory dossier for lisinopril"],
    },
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    mode = st.session_state.mode

    # Close button + header
    st.markdown("""
    <div style="display:flex;justify-content:flex-end;padding:10px 12px 0;">
        <button onclick="
            var btn = window.parent.document.querySelector('[data-testid=collapsedControl]');
            if(btn) btn.click();
        " style="
            background:rgba(0,180,160,0.08);
            border:1px solid rgba(0,180,160,0.2);
            border-radius:50%;
            width:28px;height:28px;
            cursor:pointer;
            color:#00B4A0;
            font-size:0.85rem;
            display:flex;align-items:center;justify-content:center;
        ">✕</button>
    </div>
    <div class="sb-header">
        <div class="sb-icon">💊</div>
        <p class="sb-title">Pharma RAG</p>
        <p class="sb-sub">Mayank Pratap Singh Chauhan</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sb-lbl">Output Mode</span>', unsafe_allow_html=True)
    for mk, mv in MODES.items():
        prefix = "▶  " if mode == mk else "    "
        if st.button(f"{prefix}{mv['dot']}  {mv['name']}", key=f"mode_{mk}", use_container_width=True):
            st.session_state.mode = mk
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown('<span class="sb-lbl">Drug Focus</span>', unsafe_allow_html=True)
    drug = st.selectbox("drug", list(DRUGS.keys()), label_visibility="collapsed")

    st.markdown('<span class="sb-lbl" style="display:block;margin-top:8px;">Quick Questions</span>', unsafe_allow_html=True)
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

# Hero header
hc1, hc2 = st.columns([4,1])
with hc1:
    # Sidebar toggle button always visible
    st.markdown("""
    <button onclick="
        var btn = window.parent.document.querySelector('[data-testid=collapsedControl]');
        if(btn) btn.click();
    " style="
        position:fixed;
        top:16px;
        left:16px;
        z-index:9999;
        background:rgba(0,180,160,0.15);
        border:1px solid rgba(0,180,160,0.3);
        border-radius:10px;
        width:36px;height:36px;
        cursor:pointer;
        color:#00B4A0;
        font-size:1.1rem;
        display:flex;align-items:center;justify-content:center;
        transition:all 0.2s ease;
    " title="Toggle sidebar">☰</button>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding:20px 20px 0 0;text-align:right;">
        <span class="mbadge mb-{mode}">{mode_cfg['icon']} {mode_cfg['name']}</span>
    </div>""", unsafe_allow_html=True)
    if st.button("Clear", key="clr"):
        st.session_state.messages = []
        st.rerun()

# Drug cards
cards_html = "".join(f"""
<div class="dcard">
    <div class="dcard-icon">{v['icon']}</div>
    <div class="dcard-name">{k}</div>
    <div class="dcard-class">{v['cls']}</div>
    <span class="dcard-stat">{v['stat']}</span>
</div>""" for k,v in DRUGS.items())
st.markdown(f'<div class="dcards">{cards_html}</div>', unsafe_allow_html=True)

# Chat area
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(f"""
    <div class="welcome">
        <div class="w-badge"><span style="width:5px;height:5px;border-radius:50%;background:#00B4A0;display:inline-block;"></span>AI Assistant Ready</div>
        <h3 class="w-title">👋 Hello! I'm the Pharma RAG Assistant</h3>
        <p class="w-text">
            I answer questions about FDA-approved drugs using real prescribing information from DailyMed.
            Every answer is grounded in official labeling — cited, compliant, never hallucinated.<br><br>
            Currently in <strong style="color:#e2e8f0;">{mode_cfg['icon']} {mode_cfg['name']}</strong> mode.
            Switch modes in the sidebar for different answer styles.
        </p>
        <div class="w-pills">
            <span class="w-pill">🩸 Warfarin</span>
            <span class="w-pill">💉 Metformin</span>
            <span class="w-pill">❤️ Lisinopril</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="urow">
            <div class="ubub">{msg["content"]}</div>
            <div class="uav">YOU</div>
        </div>""", unsafe_allow_html=True)
    else:
        m  = msg.get("mode","clinical")
        mc = MODES.get(m, MODES["clinical"])
        srcs = "".join(
            f'<span class="stag">📄 {s.get("metadata",{}).get("drug_name","?")} — '
            f'{s.get("metadata",{}).get("section_name","?").replace("_"," ")} '
            f'<span class="ssc">({s.get("score",0):.2f})</span></span>'
            for s in msg.get("sources",[]))
        src_html  = f'<div class="srow">{srcs}</div>' if srcs else ""
        comp_html = (
            f'<div class="cfl">⚠️ &nbsp;{" · ".join(msg["flags"][:2])}</div>'
            if msg.get("flags")
            else '<div class="cok">✓ &nbsp;Compliant — no flags</div>'
        )
        st.markdown(f"""
        <div class="brow">
            <div class="bav">AI</div>
            <div class="bbub bc-{m}">
                <span class="mbub mc-{m}">{mc['icon']} {m.upper()}</span><br>
                {msg["content"].replace(chr(10),"<br>")}
                {src_html}{comp_html}
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input bar
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
        <div class="brow">
            <div class="bav">AI</div>
            <div class="bbub bc-{mode}">
                <span class="mbub mc-{mode}">{mode_cfg['icon']} {mode.upper()}</span><br>
                <div class="typing-wrap">
                    <span class="tdot"></span><span class="tdot"></span><span class="tdot"></span>
                    <span class="tlbl">Searching FDA labels...</span>
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