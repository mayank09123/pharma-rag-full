import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load secrets from Streamlit Cloud
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        for k, v in st.secrets.items():
            os.environ.setdefault(str(k), str(v))
except:
    pass

from dotenv import load_dotenv
load_dotenv()

def ensure_loaded():
    from src.retrieval.vector_store import DrugLabelVectorStore
    from src.ingestion.fda_loader import FDALabelLoader
    store = DrugLabelVectorStore()
    if store.collection_stats()["total_chunks"] == 0:
        print("Loading FDA labels...")
        loader = FDALabelLoader()
        chunks = loader.load_multiple(["warfarin","metformin","lisinopril"])
        store.add_chunks(chunks)
        print(f"Done — {len(chunks)} chunks loaded")

ensure_loaded()
