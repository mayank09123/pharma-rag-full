"""
Ingest FDA drug labels into the vector store.
Run: python scripts/ingest.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.fda_loader import FDALabelLoader
from src.retrieval.vector_store import DrugLabelVectorStore

# ── Edit this list to ingest different drugs ──────────────────────────────
DRUGS = [
    "warfarin",
    "metformin",
    "lisinopril",
]
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  FDA Drug Label Ingestion")
    print("=" * 50)

    print(f"\nDrugs to ingest: {DRUGS}\n")

    print("Step 1: Fetching FDA labels from DailyMed...")
    loader = FDALabelLoader()
    chunks = loader.load_multiple(DRUGS)

    if not chunks:
        print("\n❌ No chunks loaded. Check your internet connection and drug names.")
        sys.exit(1)

    print(f"\n✓ Fetched {len(chunks)} total chunks\n")

    print("Step 2: Embedding and storing in ChromaDB...")
    store = DrugLabelVectorStore()
    store.add_chunks(chunks)

    stats = store.collection_stats()
    print(f"\n✓ Done! Vector store stats: {stats}")
    print("\nYou can now run: python scripts/query.py")


if __name__ == "__main__":
    main()
