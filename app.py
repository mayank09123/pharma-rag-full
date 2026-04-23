import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.retrieval.vector_store import DrugLabelVectorStore
from src.generation.rag_pipeline import PharmaRAGPipeline, OutputMode

st.set_page_config(page_title="Pharma RAG Pipeline", page_icon="💊", layout="wide")

st.markdown("""
<style>
.metric-card{background:#D4F1EE;border-radius:8px;padding:12px;text-align:center;border:1px solid #007A6E;}
.metric-val{font-size:1.8rem;font-weight:700;color:#007A6E;}
.metric-lbl{font-size:0.7rem;color:#6B7280;text-transform:uppercase;}
.flag-box{background:#FEE2E2;border-left:4px solid #B91C1C;padding:8px 12px;border-radius:4px;margin:4px 0;}
.clean-box{background:#D1FAE5;border-left:4px solid #059669;padding:8px 12px;border-radius:4px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    store = DrugLabelVectorStore()
    return store, PharmaRAGPipeline(vector_store=store)

with st.sidebar:
    st.markdown("## 💊 Pharma RAG")
    st.markdown("**Author:** Mayank Pratap Singh Chauhan\n\n**BU#:** B01098725\n\n**Advisor:** Prof. Sujoy Sikdar")
    st.divider()
    mode_label = st.selectbox("Output Mode",
        ["clinical","marketing","patient","regulatory"],
        format_func=lambda x: {"clinical":"🏥 Clinical","marketing":"📢 Marketing",
                                "patient":"👤 Patient","regulatory":"📋 Regulatory"}[x])
    st.divider()
    st.markdown("**Example Questions**")
    for ex in ["What are the contraindications for warfarin?",
               "What is the dosage for metformin?",
               "Side effects of lisinopril?",
               "Drug interactions with warfarin?",
               "Is warfarin safe during pregnancy?"]:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex
    st.divider()
    try:
        store, _ = load_pipeline()
        stats = store.collection_stats()
        st.metric("Chunks in DB", stats["total_chunks"])
    except Exception as e:
        st.error(f"Store error: {e}")

st.title("💊 FDA Drug Label RAG Pipeline")
st.caption("Grounded, compliant answers from FDA-approved labeling · Binghamton University, SUNY")
st.divider()

c1,c2,c3,c4,c5 = st.columns(5)
for col,val,lbl in [(c1,"1.00","Faithfulness"),(c2,"1.00","Precision"),
                    (c3,"0.96","Score"),(c4,"100%","Clean Rate"),(c5,"199","Chunks")]:
    col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                 f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)
st.markdown("")

question = st.text_input("Ask a question about any drug:",
    value=st.session_state.get("question",""),
    placeholder="e.g. What are the contraindications for warfarin?")

col1, col2 = st.columns([1,5])
run = col1.button("🔍 Search", type="primary", use_container_width=True)
if col2.button("🗑️ Clear"):
    st.session_state["question"] = ""
    st.rerun()

if run:
    if not question:
        st.warning("Please enter a question.")
    else:
        store, pipeline = load_pipeline()
        if store.collection_stats()["total_chunks"] == 0:
            st.error("Vector store empty. Run: python scripts/ingest.py")
        else:
            with st.spinner("Retrieving and generating..."):
                response = pipeline.query(question, mode=OutputMode(mode_label))
            st.markdown("### Answer")
            st.markdown(response.answer)
            st.markdown("### Compliance Check")
            if response.compliance_flags:
                for f in response.compliance_flags:
                    st.markdown(f'<div class="flag-box">⚠️ {f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="clean-box">✅ No compliance flags</div>',
                            unsafe_allow_html=True)
            st.markdown(f"### Sources ({len(response.sources)})")
            for i,s in enumerate(response.sources,1):
                m = s["metadata"]
                with st.expander(f"[{i}] {m.get('drug_name','?')} — "
                                 f"{m.get('section_name','?')} (score: {s['score']:.3f})"):
                    st.write(s["text"])
            with st.expander("Query Stats"):
                st.write(f"Model: {response.model} | Tokens: {response.tokens_used} | Mode: {response.mode.value}")

st.divider()
st.markdown("<center><small>Built on FDA DailyMed · GPT-4o · ChromaDB · "
            "Binghamton University · 2025</small></center>", unsafe_allow_html=True)
