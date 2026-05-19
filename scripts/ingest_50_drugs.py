import sys, os, time, argparse, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from src.ingestion.fda_loader import FDALabelLoader
from src.retrieval.vector_store import DrugLabelVectorStore

DRUG_LIBRARY = {
    "cardiovascular":    ["warfarin","aspirin","atorvastatin","metoprolol","amlodipine","lisinopril","losartan","furosemide","digoxin","clopidogrel"],
    "diabetes":          ["metformin","glipizide","sitagliptin","empagliflozin"],
    "antibiotics":       ["amoxicillin","azithromycin","ciprofloxacin","doxycycline","vancomycin"],
    "pain_inflammation": ["ibuprofen","acetaminophen","naproxen","celecoxib","prednisone"],
    "mental_health":     ["sertraline","fluoxetine","escitalopram","alprazolam","quetiapine"],
    "respiratory":       ["albuterol","fluticasone","montelukast","tiotropium"],
    "gastrointestinal":  ["omeprazole","pantoprazole","ondansetron","metoclopramide"],
    "neurology":         ["gabapentin","levetiracetam","donepezil","sumatriptan"],
    "oncology":          ["tamoxifen","methotrexate","letrozole","capecitabine"],
    "hormones":          ["levothyroxine","estradiol","hydrocortisone"],
}

seen = set()
UNIQUE_DRUGS = []
for cat, drugs in DRUG_LIBRARY.items():
    for drug in drugs:
        if drug not in seen:
            seen.add(drug)
            UNIQUE_DRUGS.append({"name": drug, "category": cat})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--drug",     type=str, default=None)
    parser.add_argument("--clear",    action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("  FDA Drug Label Ingestion — 50+ Drugs")
    print("  Mayank Pratap Singh Chauhan (B01098725)")
    print("="*60)

    if args.drug:
        drug_list = [{"name": args.drug, "category": "custom"}]
    elif args.category:
        if args.category not in DRUG_LIBRARY:
            print(f"Unknown category. Available: {', '.join(DRUG_LIBRARY.keys())}")
            sys.exit(1)
        drug_list = [{"name": d, "category": args.category} for d in DRUG_LIBRARY[args.category]]
    else:
        drug_list = UNIQUE_DRUGS

    print(f"\nIngesting {len(drug_list)} drugs...")

    if args.clear:
        import shutil
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            print("✓ Cleared existing database")

    loader  = FDALabelLoader()
    store   = DrugLabelVectorStore()
    success = []
    failed  = []

    for i, d in enumerate(drug_list, 1):
        name = d["name"]; cat = d["category"]
        print(f"\n[{i}/{len(drug_list)}] {name.upper()} ({cat})")
        try:
            results = loader.search_drugs(name, limit=1)
            if not results:
                print(f"  ✗ Not found"); failed.append(name); continue
            set_id = results[0]["setid"]
            chunks = loader.load_label_by_setid(set_id)
            for chunk in chunks:
                chunk.metadata["category"] = cat
            store.add_chunks(chunks)
            print(f"  ✓ {len(chunks)} chunks stored")
            success.append({"name": name, "chunks": len(chunks)})
        except Exception as e:
            print(f"  ✗ Error: {e}"); failed.append(name)
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"  ✓ Success: {len(success)} drugs")
    print(f"  ✗ Failed:  {len(failed)} drugs")
    print(f"  📦 Database: {store.collection_stats()['total_chunks']} total chunks")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print("\nRun: python scripts/query.py")

if __name__ == "__main__":
    main()
