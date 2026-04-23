"""
Query the RAG pipeline interactively.
Run: python scripts/query.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.vector_store import DrugLabelVectorStore
from src.generation.rag_pipeline import PharmaRAGPipeline, OutputMode

# ── Edit these to try different queries ──────────────────────────────────
QUESTION = "What are the contraindications for warfarin?"
MODE = OutputMode.CLINICAL   # CLINICAL | MARKETING | PATIENT | REGULATORY
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Pharma RAG Pipeline — Query")
    print("=" * 50)

    print("\nConnecting to vector store...")
    store = DrugLabelVectorStore()
    stats = store.collection_stats()
    print(f"Vector store: {stats['total_chunks']} chunks loaded")

    if stats["total_chunks"] == 0:
        print("\n❌ Vector store is empty. Run python scripts/ingest.py first.")
        sys.exit(1)

    print("\nInitializing pipeline...")
    pipeline = PharmaRAGPipeline(vector_store=store)

    print(f"\nQuery   : {QUESTION}")
    print(f"Mode    : {MODE.value.upper()}")
    print("\nRunning RAG pipeline...")

    response = pipeline.query(QUESTION, mode=MODE)

    print("\n" + "=" * 50)
    print("  ANSWER")
    print("=" * 50)
    print(response.answer)

    print("\n" + "=" * 50)
    print(f"  SOURCES ({len(response.sources)})")
    print("=" * 50)
    for i, s in enumerate(response.sources, 1):
        meta = s["metadata"]
        print(f"  [{i}] Score: {s['score']:.3f} | {meta.get('drug_name','?')} — {meta.get('section_name','?')}")

    print("\n" + "=" * 50)
    print("  COMPLIANCE FLAGS")
    print("=" * 50)
    if response.compliance_flags:
        for flag in response.compliance_flags:
            print(f"  ⚠️  {flag}")
    else:
        print("  ✓ No compliance flags")

    print(f"\nTokens used: {response.tokens_used}")


if __name__ == "__main__":
    main()
