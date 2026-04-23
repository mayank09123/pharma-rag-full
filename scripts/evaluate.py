"""
Run evaluation on the RAG pipeline.
Run: python scripts/evaluate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import json
from src.retrieval.vector_store import DrugLabelVectorStore
from src.generation.rag_pipeline import PharmaRAGPipeline, OutputMode
from src.evaluation.evaluator import RAGEvaluator

TEST_QUESTIONS = [
    {"query": "What are the contraindications for warfarin?", "mode": "clinical"},
    {"query": "What is the recommended dosage for metformin?", "mode": "clinical"},
    {"query": "What are the side effects of lisinopril?", "mode": "patient"},
    {"query": "Write a brief about warfarin for physicians", "mode": "marketing"},
    {"query": "What drug interactions should be monitored with warfarin?", "mode": "clinical"},
]


def main():
    print("=" * 50)
    print("  RAG Evaluation Suite")
    print("=" * 50)

    store = DrugLabelVectorStore()
    if store.collection_stats()["total_chunks"] == 0:
        print("\n❌ Vector store is empty. Run python scripts/ingest.py first.")
        sys.exit(1)

    pipeline = PharmaRAGPipeline(vector_store=store)
    evaluator = RAGEvaluator(use_llm_judge=False)  # set True to use LLM judge (costs tokens)

    results = []
    for i, case in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {case['query'][:60]}...")
        mode = OutputMode(case["mode"])
        response = pipeline.query(case["query"], mode=mode)
        result = evaluator.evaluate_response(response, query_id=f"tc_{i:03d}")
        results.append(result)
        print(f"  Overall score: {result.overall_score:.3f} | Faithfulness: {result.faithfulness_score:.3f} | Flags: {result.flag_count}")

    report = evaluator.generate_report(results)

    print("\n" + "=" * 50)
    print("  EVALUATION REPORT")
    print("=" * 50)
    for k, v in report.items():
        print(f"  {k}: {v}")

    output_path = "eval_results.json"
    with open(output_path, "w") as f:
        json.dump({"report": report, "results": [vars(r) for r in results]}, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
