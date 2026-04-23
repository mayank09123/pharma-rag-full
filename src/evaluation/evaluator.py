"""
RAG Evaluation Framework
Measures factual accuracy, retrieval quality, compliance, and user feedback.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.generation.rag_pipeline import RAGResponse


@dataclass
class EvalResult:
    query_id: str
    query: str
    answer: str
    mode: str
    retrieval_precision: float
    mean_reciprocal_rank: float
    faithfulness_score: float
    answer_relevance: float
    citation_coverage: float
    compliance_flags: list
    flag_count: int
    human_score: Optional[float] = None
    human_notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def overall_score(self) -> float:
        compliance_score = max(0.0, 1.0 - 0.2 * self.flag_count)
        return (
            0.25 * self.retrieval_precision
            + 0.35 * self.faithfulness_score
            + 0.20 * self.answer_relevance
            + 0.10 * self.citation_coverage
            + 0.10 * compliance_score
        )


class RAGEvaluator:
    """
    Evaluate RAG pipeline responses.

    Usage:
        evaluator = RAGEvaluator(use_llm_judge=False)  # fast mode
        result = evaluator.evaluate_response(rag_response)
        print(result.overall_score)
    """

    def __init__(self, use_llm_judge: bool = False):
        self._use_llm = use_llm_judge
        if use_llm_judge:
            import os
            from openai import OpenAI
            self._llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def evaluate_response(
        self,
        response: RAGResponse,
        relevant_set_ids: Optional[list] = None,
        query_id: Optional[str] = None,
    ) -> EvalResult:
        sources = response.sources
        answer = response.answer

        retrieval_precision = self._retrieval_precision(sources, relevant_set_ids)
        mrr = self._mrr(sources, relevant_set_ids)

        citations_found = len(re.findall(r"\[SOURCE \d+\]", answer))
        citation_coverage = min(1.0, citations_found / max(len(sources), 1))

        if self._use_llm:
            faithfulness = self._llm_faithfulness(answer, sources)
            relevance = self._llm_relevance(response.query, answer)
        else:
            faithfulness = self._heuristic_faithfulness(answer, sources)
            relevance = min(1.0, citation_coverage + 0.4)

        return EvalResult(
            query_id=query_id or f"q_{abs(hash(response.query)) % 10000:04d}",
            query=response.query,
            answer=answer,
            mode=response.mode.value,
            retrieval_precision=retrieval_precision,
            mean_reciprocal_rank=mrr,
            faithfulness_score=faithfulness,
            answer_relevance=relevance,
            citation_coverage=citation_coverage,
            compliance_flags=response.compliance_flags,
            flag_count=len(response.compliance_flags),
        )

    def generate_report(self, results: list) -> dict:
        if not results:
            return {}

        def avg(vals):
            return round(statistics.mean(vals), 3)

        return {
            "n_queries": len(results),
            "avg_overall_score": avg([r.overall_score for r in results]),
            "avg_retrieval_precision": avg([r.retrieval_precision for r in results]),
            "avg_mrr": avg([r.mean_reciprocal_rank for r in results]),
            "avg_faithfulness": avg([r.faithfulness_score for r in results]),
            "avg_answer_relevance": avg([r.answer_relevance for r in results]),
            "avg_citation_coverage": avg([r.citation_coverage for r in results]),
            "total_compliance_flags": sum(r.flag_count for r in results),
            "pct_clean_responses": round(
                sum(1 for r in results if r.flag_count == 0) / len(results), 3
            ),
        }

    def _retrieval_precision(self, sources: list, relevant_ids: Optional[list]) -> float:
        if not relevant_ids:
            return 1.0
        hits = sum(1 for s in sources if s["metadata"].get("set_id") in relevant_ids)
        return hits / max(len(sources), 1)

    def _mrr(self, sources: list, relevant_ids: Optional[list]) -> float:
        if not relevant_ids:
            return 1.0
        for rank, s in enumerate(sources, 1):
            if s["metadata"].get("set_id") in relevant_ids:
                return 1.0 / rank
        return 0.0

    def _heuristic_faithfulness(self, answer: str, sources: list) -> float:
        answer_tokens = set(answer.lower().split())
        source_tokens = set()
        for s in sources:
            source_tokens.update(s["text"].lower().split())
        if not answer_tokens:
            return 0.0
        overlap = len(answer_tokens & source_tokens) / len(answer_tokens)
        return round(min(overlap * 2, 1.0), 3)

    def _llm_faithfulness(self, answer: str, sources: list) -> float:
        source_text = "\n\n".join(s["text"][:400] for s in sources[:3])
        prompt = f"""Rate faithfulness 0.0-1.0. Does this answer only use facts from these sources?

SOURCES:
{source_text}

ANSWER:
{answer}

Respond with JSON only: {{"score": <float>}}"""
        resp = self._llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return float(json.loads(resp.choices[0].message.content)["score"])

    def _llm_relevance(self, question: str, answer: str) -> float:
        prompt = f"""Rate 0.0-1.0: does this answer address the question?

QUESTION: {question}
ANSWER: {answer}

Respond with JSON only: {{"score": <float>}}"""
        resp = self._llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return float(json.loads(resp.choices[0].message.content)["score"])
