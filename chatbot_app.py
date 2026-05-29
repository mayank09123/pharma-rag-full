"""
Pharma RAG Chatbot
Author: Mayank Pratap Singh Chauhan (B01098725)

"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# Load secrets from Streamlit Cloud first
try:
    if hasattr(st, 'secrets'):
        for k, v in st.secrets.items():
            os.environ[str(k)] = str(v)
except:
    pass

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.vector_store import DrugLabelVectorStore
from src.generation.rag_pipeline import PharmaRAGPipeline, OutputMode

st.set_page_config(page_title="Pharma RAG", page_icon="💊", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
*{font-family:'Inter',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stSidebar"]{background:#111827!important;}
section[data-testid="stSidebar"] > div{background:#111827!important;padding:16px!important;}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label{color:#f1f5f9!important;}
section[data-testid="stSidebar"] .stButton>button{
    background:#1f2937!important;border:1px solid #374151!important;
    color:#f1f5f9!important;border-radius:8px!important;
    padding:8px 12px!important;font-size:0.82rem!important;
    text-align:left!important;width:100%!important;margin-bottom:4px!important;}
section[data-testid="stSidebar"] .stButton>button:hover{background:#374151!important;border-color:#00B4A0!important;}
[data-testid="stMetric"]{background:#1f2937!important;border-radius:10px!important;padding:10px!important;border:1px solid #374151!important;}
[data-testid="stMetricValue"]{color:#00B4A0!important;font-size:1.5rem!important;font-weight:600!important;}
[data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:0.7rem!important;}
.main .block-container{background:#0f172a!important;min-height:100vh!important;padding:20px 28px!important;}
.user-row{display:flex;justify-content:flex-end;align-items:flex-start;gap:10px;margin:10px 0;}
.user-av{width:30px;height:30px;border-radius:50%;background:#374151;display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:600;color:#94a3b8;flex-shrink:0;}
.user-bub{background:#1d4ed8;border-radius:18px 18px 4px 18px;padding:10px 16px;font-size:0.88rem;color:white;max-width:70%;line-height:1.6;}
.bot-row{display:flex;justify-content:flex-start;align-items:flex-start;gap:10px;margin:10px 0;}
.bot-av{width:30px;height:30px;border-radius:50%;background:#0f3d2e;display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:700;color:#00B4A0;flex-shrink:0;border:1px solid #1D9E75;}
.bot-bub{background:#1e293b;border:1px solid #2d3d50;border-radius:4px 18px 18px 18px;padding:14px 16px;font-size:0.88rem;color:#e2e8f0;max-width:85%;line-height:1.7;}
.mbadge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.65rem;font-weight:700;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em;}
.mb-clinical{background:#0f3d2e;color:#00B4A0;}
.mb-marketing{background:#3d2d00;color:#EF9F27;}
.mb-patient{background:#0a1f40;color:#85B7EB;}
.mb-regulatory{background:#1a0f40;color:#AFA9EC;}
.src-row{display:flex;flex-wrap:wrap;gap:4px;margin-top:10px;padding-top:8px;border-top:1px solid #2d3d50;}
.src-tag{background:#0f172a;border:1px solid #2d3d50;border-radius:20px;padding:2px 10px;font-size:0.67rem;color:#94a3b8;}
.src-sc{color:#00B4A0;font-weight:600;}
.comp-ok{color:#4ade80;font-size:0.7rem;margin-top:5px;}
.comp-fl{color:#f87171;font-size:0.7rem;margin-top:5px;}
.typing-dot{display:inline-block;width:6px;height:6px;background:#00B4A0;border-radius:50%;margin:0 2px;animation:td 1.2s infinite;}
.typing-dot:nth-child(2){animation-delay:0.2s;}.typing-dot:nth-child(3){animation-delay:0.4s;}
@keyframes td{0%,60%,100%{transform:translateY(0);opacity:0.4;}30%{transform:translateY(-4px);opacity:1;}}
.stTextInput input{background:#1e293b!important;border:1px solid #374151!important;border-radius:24px!important;color:#f1f5f9!important;padding:11px 20px!important;font-size:0.9rem!important;}
.stTextInput input:focus{border-color:#007A6E!important;box-shadow:none!important;}
.stTextInput input::placeholder{color:#475569!important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    # Auto ingest if database is empty
    store = DrugLabelVectorStore(use_rerank=False)
    if store.collection_stats()["total_chunks"] == 0:
        from src.ingestion.fda_loader import FDALabelLoader
        loader = FDALabelLoader()
        chunks = loader.load_multiple(["warfarin","metformin","lisinopril"])
        store.add_chunks(chunks)
    return store, PharmaRAGPipeline(vector_store=store)

for k,v in {"messages":[],"input_key":0,"mode":"clinical"}.items():
    if k not in st.session_state: st.session_state[k] = v

MODES = {
    "clinical":   {"dot":"🟢","name":"Clinical (HCP)"},
    "marketing":  {"dot":"🟡","name":"Marketing"},
    "patient":    {"dot":"🔵","name":"Patient"},
    "regulatory": {"dot":"🟣","name":"Regulatory"},
}
ALL_QUESTIONS = {
    "clinical": {
        "Warfarin":   ["What are the contraindications for warfarin?","What are the side effects of warfarin?","What drug interactions does warfarin have?","What is the dosage for warfarin?","How does warfarin work?","Is warfarin safe during pregnancy?","What monitoring is required for warfarin?"],
        "Metformin":  ["What is the recommended dose of metformin?","What are the side effects of metformin?","How does metformin work?","Is metformin safe in kidney disease?","Can metformin cause lactic acidosis?"],
        "Lisinopril": ["What are the contraindications for lisinopril?","What is the dosage for lisinopril?","What are the side effects of lisinopril?","How does lisinopril work?","Can lisinopril cause a cough?"],
    },
    "marketing": {
        "Warfarin":   ["Write a physician brief for warfarin","What is the approved indication for warfarin?","What safety info must be included for warfarin?"],
        "Metformin":  ["Write a marketing brief for metformin","What clinical evidence supports metformin?"],
        "Lisinopril": ["Promotional overview of lisinopril","What are the benefits of lisinopril?"],
    },
    "patient": {
        "Warfarin":   ["What is warfarin used for?","What foods should I avoid with warfarin?","Can I drink alcohol with warfarin?","Is warfarin safe in pregnancy?","What if I miss a warfarin dose?"],
        "Metformin":  ["What does metformin do?","Will metformin make me feel sick?","What if I forget metformin?"],
        "Lisinopril": ["Why do people take lisinopril?","How long does lisinopril take to work?","Is lisinopril safe during breastfeeding?"],
    },
    "regulatory": {
        "Warfarin":   ["Regulatory summary of warfarin","What is the verbatim indication for warfarin?","What are the boxed warnings for warfarin?"],
        "Metformin":  ["FDA approved indication for metformin","Regulatory summary of metformin"],
        "Lisinopril": ["Safety profile of lisinopril","Regulatory dossier summary for lisinopril"],
    },
}

with st.sidebar:
    st.markdown("### 💊 Pharma RAG")
    st.markdown("Mayank (B01098725)")
    st.divider()
    st.markdown("**OUTPUT MODE**")
    mode = st.session_state.mode
    for mk, mv in MODES.items():
        prefix = "→ " if mode == mk else "   "
        if st.button(f"{prefix}{mv['dot']} {mv['name']}", key=f"mode_{mk}", use_container_width=True):
            st.session_state.mode = mk
            st.session_state.messages = []
            st.rerun()
    st.divider()
    drug_tab = st.selectbox("Drug", ["Warfarin","Metformin","Lisinopril"], label_visibility="collapsed")
    questions = ALL_QUESTIONS.get(mode, {}).get(drug_tab, [])
    st.markdown(f"**{drug_tab} — {MODES[mode]['name']}**")
    for i, q in enumerate(questions):
        if st.button(q, key=f"q_{mode}_{drug_tab}_{i}", use_container_width=True):
            st.session_state["pending"] = q
    st.divider()
    try:
        store, _ = load_pipeline()
        stats = store.collection_stats()
        c1, c2 = st.columns(2)
        c1.metric("Chunks", stats["total_chunks"])
        c2.metric("Score",  "0.96")
    except Exception as e:
        st.error(f"Error: {e}")
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

mode     = st.session_state.mode
mode_cfg = MODES[mode]
hc1, hc2 = st.columns([5,1])
with hc1:
    st.markdown("<h3 style='color:white;margin:0;'>Pharma RAG Chatbot</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;font-size:0.78rem;margin:2px 0 0;'>FDA DailyMed · GPT-4o · ChromaDB · Binghamton University</p>", unsafe_allow_html=True)
with hc2:
    st.markdown(f"<div style='text-align:right;padding-top:4px;'><span style='background:#1e293b;border:1px solid #374151;border-radius:20px;padding:5px 14px;font-size:0.78rem;color:#e2e8f0;'>{mode_cfg['dot']} {mode_cfg['name']}</span></div>", unsafe_allow_html=True)
    if st.button("Clear", key="clear_main"):
        st.session_state.messages = []
        st.rerun()
st.divider()

if not st.session_state.messages:
    st.markdown(f"""<div class="bot-row">
        <div class="bot-av">AI</div>
        <div class="bot-bub">
            <span class="mbadge mb-{mode}">{mode.upper()}</span><br>
            👋 <b>Hello! I'm the Pharma RAG Assistant.</b><br><br>
            I answer questions about <b>Warfarin · Metformin · Lisinopril</b>
            using real FDA prescribing information.<br><br>
            Currently in <b>{mode_cfg['dot']} {mode_cfg['name']}</b> mode.<br><br>
            Pick a drug from the sidebar then click any question — or type your own!
        </div></div>""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""<div class="user-row">
            <div class="user-bub">{msg["content"]}</div>
            <div class="user-av">You</div>
        </div>""", unsafe_allow_html=True)
    else:
        m  = msg.get("mode","clinical")
        mc = MODES.get(m, MODES["clinical"])
        srcs = "".join(
            f'<span class="src-tag">📄 {s.get("metadata",{}).get("drug_name","?")} — '
            f'{s.get("metadata",{}).get("section_name","?")} '
            f'<span class="src-sc">({s.get("score",0):.2f})</span></span>'
            for s in msg.get("sources",[]))
        src_html  = f'<div class="src-row">{srcs}</div>' if srcs else ""
        comp_html = (f'<div class="comp-fl">⚠️ {" · ".join(msg["flags"][:2])}</div>'
                     if msg.get("flags")
                     else '<div class="comp-ok">✓ Compliant — no flags</div>')
        st.markdown(f"""<div class="bot-row">
            <div class="bot-av">AI</div>
            <div class="bot-bub">
                <span class="mbadge mb-{m}">{m.upper()}</span><br>
                {msg["content"].replace(chr(10),"<br>")}
                {src_html}{comp_html}
            </div></div>""", unsafe_allow_html=True)

pending = st.session_state.pop("pending", None)
ci, cb  = st.columns([11, 1])
with ci:
    user_input = st.text_input("q", value=pending or "",
        placeholder="Ask about any drug...",
        label_visibility="collapsed",
        key=f"inp_{st.session_state.input_key}")
with cb:
    send = st.button("➤", type="primary", use_container_width=True)

question = (pending or user_input).strip()
if (send or pending) and question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.empty():
        st.markdown("""<div class="bot-row">
            <div class="bot-av">AI</div>
            <div class="bot-bub">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                &nbsp;<small style="color:#475569">Searching FDA labels...</small>
            </div></div>""", unsafe_allow_html=True)
        time.sleep(0.8)
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
