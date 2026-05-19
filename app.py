import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from src.retrieval.vector_store import DrugLabelVectorStore
from src.generation.rag_pipeline import PharmaRAGPipeline, OutputMode


# ─────────────────────────────────────────────────────────────
# UI CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma RAG Pipeline",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
<style>
.metric-card{background:#D4F1EE;border-radius:8px;padding:12px;text-align:center;border:1px solid #007A6E;}
.metric-val{font-size:1.8rem;font-weight:700;color:#007A6E;}
.metric-lbl{font-size:0.7rem;color:#6B7280;text-transform:uppercase;}
.flag-box{background:#FEE2E2;border-left:4px solid #B91C1C;padding:8px 12px;border-radius:4px;margin:4px 0;}
.clean-box{background:#D1FAE5;border-left:4px solid #059669;padding:8px 12px;border-radius:4px;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PIPELINE LOADER (FIXED)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    store = DrugLabelVectorStore()

    # 🔴 FIX: Auto-ingest if DB is empty
    if store.collection_stats()["total_chunks"] == 0:
        st.warning("Vector DB empty. Building index (first run only)...")

        from scripts.ingest import main as ingest_main
        ingest_main()

        # reload after ingestion
        store = DrugLabelVectorStore()

    pipeline = PharmaRAGPipeline(vector_store=store)
    return store, pipeline


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 Pharma RAG")
    st.markdown("**Author:** Mayank Pratap Singh Chauhan\n\n**BU#:** B01098725\n\n**Advisor:** Prof. Sujoy Sikdar")
    st.divider()

    mode_label = st.selectbox(
        "Output Mode",
        ["clinical", "marketing", "patient", "regulatory"],
        format_func=lambda x: {
            "clinical": "🏥 Clinical",
            "marketing": "📢 Marketing",
            "patient": "👤 Patient",
            "regulatory": "📋 Regulatory"
        }[x]
    )

    st.divider()
    st.markdown("**Example Questions**")

    examples = [
        "What are the contraindications for warfarin?",
        "What is the dosage for metformin?",
        "Side effects of lisinopril?",
        "Drug interactions with warfarin?",
        "Is warfarin safe during pregnancy?"
    ]

    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

    st.divider()

    try:
        store, _ = load_pipeline()
        stats = store.collection_stats()
        st.metric("Chunks in DB", stats["total_chunks"])
    except Exception as e:
        st.error(f"Store error: {e}")


# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────
st.title("💊 FDA Drug Label RAG Pipeline")
st.caption("Grounded, compliant answers from FDA-approved labeling")
st.divider()


# metrics row
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    ("1.00", "Faithfulness"),
    ("1.00", "Precision"),
    ("0.96", "Score"),
    ("100%", "Clean Rate"),
    ("199", "Chunks")
]

for col, (val, lbl) in zip([c1, c2, c3, c4, c5], metrics):
    col.markdown(
        f'<div class="metric-card"><div class="metric-val">{val}</div>'
        f'<div class="metric-lbl">{lbl}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("")


# ─────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────
question = st.text_input(
    "Ask a question about any drug:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What are the contraindications for warfarin?"
)

col1, col2 = st.columns([1, 5])

run = col1.button("🔍 Search", type="primary", use_container_width=True)

if col2.button("🗑️ Clear"):
    st.session_state["question"] = ""
    st.rerun()


# ─────────────────────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────────────────────
if run:
    if not question:
        st.warning("Please enter a question.")
    else:
        store, pipeline = load_pipeline()

        stats = store.collection_stats()

        if stats["total_chunks"] == 0:
            st.error("Vector store still empty. Run: python scripts/ingest.py")
        else:
            with st.spinner("Retrieving and generating..."):
                response = pipeline.query(
                    question,
                    mode=OutputMode(mode_label)
                )

            st.markdown("### Answer")
            st.markdown(response.answer)

            st.markdown("### Compliance Check")

            if response.compliance_flags:
                for f in response.compliance_flags:
                    st.markdown(
                        f'<div class="flag-box">⚠️ {f}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    '<div class="clean-box">✅ No compliance flags</div>',
                    unsafe_allow_html=True
                )

            st.markdown(f"### Sources ({len(response.sources)})")

            for i, s in enumerate(response.sources, 1):
                m = s["metadata"]
                with st.expander(
                    f"[{i}] {m.get('drug_name','?')} — "
                    f"{m.get('section_name','?')} (score: {s['score']:.3f})"
                ):
                    st.write(s["text"])

            with st.expander("Query Stats"):
                st.write(
                    f"Model: {response.model} | "
                    f"Tokens: {response.tokens_used} | "
                    f"Mode: {response.mode.value}"
                )


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Built on FDA DailyMed · GPT-4o · ChromaDB · "
    " · 2025</small></center>",
    unsafe_allow_html=True
)