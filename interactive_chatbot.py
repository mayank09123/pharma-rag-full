import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from src.retrieval.vector_store import DrugLabelVectorStore
from src.generation.rag_pipeline import PharmaRAGPipeline, OutputMode

st.set_page_config(page_title="Pharma RAG", page_icon="💊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
*{font-family:'Inter',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stSidebar"]{background:#1a1a2e!important;border-right:1px solid #2d2d4e!important;}
section[data-testid="stSidebar"] *{color:#e2e8f0!important;}
section[data-testid="stSidebar"] .stButton button{background:#2d2d4e!important;border:1px solid #3d3d5e!important;color:#e2e8f0!important;border-radius:8px!important;text-align:left!important;font-size:0.82rem!important;}
section[data-testid="stSidebar"] .stButton button:hover{background:#3d3d5e!important;border-color:#007A6E!important;}
section[data-testid="stSidebar"] .stButton button[kind="primary"]{background:#007A6E!important;border-color:#007A6E!important;color:white!important;}
section[data-testid="stSidebar"] .stSelectbox select{background:#2d2d4e!important;}
section[data-testid="stSidebar"] [data-testid="stMetricValue"]{color:#00B4A0!important;font-size:1.4rem!important;}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:0.7rem!important;}
.user-bubble{background:linear-gradient(135deg,#0B1F3A,#007A6E);border-radius:16px 16px 4px 16px;padding:12px 16px;margin:8px 0 8px auto;max-width:78%;font-size:0.9rem;color:white;text-align:right;display:block;box-shadow:0 2px 8px rgba(0,122,110,0.3);}
.bot-bubble{background:#1e2a3a;border:1px solid #2d3d50;border-left:3px solid #007A6E;border-radius:4px 16px 16px 16px;padding:14px 16px;margin:8px 8px 8px 0;max-width:90%;font-size:0.9rem;color:#e2e8f0;line-height:1.7;}
.bot-marketing{border-left-color:#C9A838!important;}
.bot-patient{border-left-color:#378ADD!important;}
.bot-regulatory{border-left-color:#7F77DD!important;}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.68rem;font-weight:600;margin-bottom:8px;letter-spacing:0.05em;}
.b-clinical{background:#0F3d2e;color:#00B4A0;}
.b-marketing{background:#3d2e0a;color:#EF9F27;}
.b-patient{background:#0a1f3d;color:#85B7EB;}
.b-regulatory{background:#1f1040;color:#AFA9EC;}
.source-row{margin-top:10px;padding-top:8px;border-top:1px solid #2d3d50;display:flex;flex-wrap:wrap;gap:4px;}
.source-tag{display:inline-block;background:#0d1b2a;border:1px solid #2d3d50;border-radius:20px;padding:2px 10px;font-size:0.68rem;color:#94a3b8;}
.src-score{color:#00B4A0;font-weight:600;}
.comp{margin-top:8px;font-size:0.72rem;}
.clean{color:#4ade80;}.flagged{color:#f87171;}
.typing-dot{display:inline-block;width:7px;height:7px;background:#007A6E;border-radius:50%;margin:0 2px;animation:bounce 1.2s infinite;}
.typing-dot:nth-child(2){animation-delay:0.2s;}.typing-dot:nth-child(3){animation-delay:0.4s;}
@keyframes bounce{0%,60%,100%{transform:translateY(0);}30%{transform:translateY(-5px);}}
.stTextInput input{border-radius:24px!important;border:1px solid #2d3d50!important;padding:10px 20px!important;font-size:0.9rem!important;background:#1e2a3a!important;color:#e2e8f0!important;}
.stTextInput input:focus{border-color:#007A6E!important;box-shadow:0 0 0 2px rgba(0,122,110,0.2)!important;}
.stTextInput input::placeholder{color:#64748b!important;}
div[data-testid="stVerticalBlock"]{background:transparent!important;}
.main-header{padding:16px 20px 0;border-bottom:1px solid #2d3d50;margin-bottom:0;}
.chat-badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:0.78rem;font-weight:600;}
.cb-clinical{background:#0F3d2e;color:#00B4A0;}
.cb-marketing{background:#3d2e0a;color:#EF9F27;}
.cb-patient{background:#0a1f3d;color:#85B7EB;}
.cb-regulatory{background:#1f1040;color:#AFA9EC;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    store = DrugLabelVectorStore(use_rerank=False)
    return store, PharmaRAGPipeline(vector_store=store)

for k,v in {"messages":[],"input_key":0,"mode":"clinical"}.items():
    if k not in st.session_state: st.session_state[k] = v

MODES = {
    "clinical":   {"icon":"🏥","name":"Clinical (HCP)", "dot":"#1D9E75"},
    "marketing":  {"icon":"📢","name":"Marketing",      "dot":"#BA7517"},
    "patient":    {"icon":"👤","name":"Patient",        "dot":"#378ADD"},
    "regulatory": {"icon":"📋","name":"Regulatory",     "dot":"#7F77DD"},
}
QUICK = {
    "clinical":   ["Warfarin contraindications","Metformin dosage","Lisinopril side effects","Warfarin interactions","Warfarin in pregnancy","How metformin works"],
    "marketing":  ["Physician brief for warfarin","Marketing summary metformin","Promotional overview lisinopril"],
    "patient":    ["Is warfarin safe in pregnancy?","Miss metformin dose?","Alcohol with warfarin?","Foods avoid with warfarin?"],
    "regulatory": ["Regulatory summary of warfarin","FDA indication for metformin","Safety profile of lisinopril"],
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Pharma RAG")
    st.markdown("Mayank (B01098725)")
    st.divider()

    st.markdown("**OUTPUT MODE**")
    for mk, mv in MODES.items():
        active = st.session_state.mode == mk
        label  = f"● {mv['name']}" if active else f"○ {mv['name']}"
        if st.button(label, key=f"m_{mk}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.mode = mk
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown("**QUICK QUESTIONS**")
    for i, q in enumerate(QUICK[st.session_state.mode]):
        if st.button(q, key=f"qq_{i}_{st.session_state.mode}", use_container_width=True):
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
    drug_focus = st.selectbox("Drug Focus", ["All drugs","Warfarin","Metformin","Lisinopril"])
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── MAIN AREA ─────────────────────────────────────────────────────────────
mode    = st.session_state.mode
mode_cfg = MODES[mode]

# Header
hcol1, hcol2 = st.columns([3,1])
with hcol1:
    st.markdown(f"### Pharma RAG Chatbot")
    st.caption("FDA DailyMed · GPT-4o · ChromaDB")
with hcol2:
    badge_html = f'<span class="chat-badge cb-{mode}">{mode_cfg["icon"]} {mode_cfg["name"]}</span>'
    st.markdown(badge_html, unsafe_allow_html=True)
st.divider()

# Welcome / chat messages
if not st.session_state.messages:
    st.markdown(f"""<div class="bot-bubble bot-{mode}">
        <span class="badge b-{mode}">{mode_cfg['icon']} {mode.upper()}</span><br>
        👋 <b>Hello! I'm the Pharma RAG Assistant.</b><br><br>
        I answer questions about <b>Warfarin · Metformin · Lisinopril</b>
        using real FDA prescribing information.<br><br>
        Currently in <b>{mode_cfg['icon']} {mode_cfg['name']}</b> mode.<br><br>
        Click a quick question in the sidebar or type below!
    </div>""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True)
    else:
        m   = msg.get("mode", "clinical")
        mc  = MODES.get(m, MODES["clinical"])
        cls = f"bot-{m}" if m != "clinical" else ""
        srcs = "".join(
            f'<span class="source-tag">📄 '
            f'{s.get("metadata",{}).get("drug_name","?")} — '
            f'{s.get("metadata",{}).get("section_name","?")} '
            f'<span class="src-score">({s.get("score",0):.2f})</span></span>'
            for s in msg.get("sources",[]))
        src_html  = f'<div class="source-row">{srcs}</div>' if srcs else ""
        comp_html = (
            f'<div class="comp flagged">⚠️ {" · ".join(msg["flags"][:2])}</div>'
            if msg.get("flags")
            else '<div class="comp clean">✓ Compliant — no flags</div>'
        )
        st.markdown(f"""<div class="bot-bubble {cls}">
            <span class="badge b-{m}">{mc['icon']} {m.upper()}</span><br>
            {msg["content"].replace(chr(10),"<br>")}
            {src_html}{comp_html}
        </div>""", unsafe_allow_html=True)

# Input bar
pending = st.session_state.pop("pending", None)
ci, cb  = st.columns([9, 1])
with ci:
    user_input = st.text_input("q",
        value=pending or "",
        placeholder=f"Ask about any drug...",
        label_visibility="collapsed",
        key=f"inp_{st.session_state.input_key}")
with cb:
    send = st.button("➤", type="primary", use_container_width=True)

question = (pending or user_input).strip()

if (send or pending) and question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.empty():
        st.markdown("""<div class="bot-bubble">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            &nbsp;<small style="color:#64748b">Searching FDA labels...</small>
        </div>""", unsafe_allow_html=True)
        time.sleep(0.8)
    try:
        store, pipeline = load_pipeline()
        drug_filter = None if drug_focus == "All drugs" else drug_focus.lower()
        response = pipeline.query(question,
            mode=OutputMode(mode),
            drug_filter=drug_filter)
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
