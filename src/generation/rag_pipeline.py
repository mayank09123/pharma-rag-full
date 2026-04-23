"""
RAG Generation Pipeline
Combines retrieved FDA content with LLM generation,
with compliance-aware prompting for pharma marketing use cases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.retrieval.vector_store import DrugLabelVectorStore


class OutputMode(str, Enum):
    CLINICAL    = "clinical"
    MARKETING   = "marketing"
    PATIENT     = "patient"
    REGULATORY  = "regulatory"


SYSTEM_PROMPTS = {
    OutputMode.CLINICAL: """You are a medical information specialist assistant.
Your responses must be:
- Grounded exclusively in the retrieved FDA-approved labeling provided
- Accurate, precise, and use appropriate clinical terminology
- Include relevant contraindications, warnings, and drug interactions when pertinent
- Always cite the section of the label using [SOURCE N] inline
- Never extrapolate beyond what is stated in the retrieved content

If the retrieved content does not answer the question, say so clearly.""",

    OutputMode.MARKETING: """You are a pharmaceutical marketing writer.
Your responses must be:
- Consistent with FDA-approved labeling (retrieved content is your source of truth)
- Balanced: include indication, relevant safety information, and fair balance language
- Compelling but never exaggerating efficacy beyond what the label supports
- Flagged with [REVIEW REQUIRED] if any claim needs medical/legal review
- Compliant with OPDP promotional guidelines

Never make efficacy claims not supported by the retrieved label content.""",

    OutputMode.PATIENT: """You are a patient education specialist.
Your responses must be:
- Written at a 6th-grade reading level
- Based only on the FDA-approved labeling content provided
- Empathetic, clear, and actionable
- Include when to contact a healthcare provider
- Always end with: "This information is not a substitute for professional medical advice."

Always cite sources using [SOURCE N] inline.""",

    OutputMode.REGULATORY: """You are a regulatory affairs specialist.
Summarize the retrieved labeling content in a structured regulatory format.
Include: approved indication, key clinical data, safety profile, and labeling sections referenced.
Be precise and cite section codes where available using [SOURCE N].""",
}


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list
    mode: OutputMode
    model: str
    tokens_used: int
    compliance_flags: list


class PharmaRAGPipeline:
    """
    End-to-end RAG pipeline for pharmaceutical queries.

    Example:
        pipeline = PharmaRAGPipeline()
        response = pipeline.query(
            "What are the contraindications for warfarin?",
            mode=OutputMode.CLINICAL
        )
        print(response.answer)
    """

    def __init__(
        self,
        vector_store: Optional[DrugLabelVectorStore] = None,
        model: str = "gpt-4o",
        k_retrieve: int = 5,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Run: pip install openai")

        self._llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._store = vector_store or DrugLabelVectorStore()
        self._model = model
        self._k = k_retrieve

    def query(
        self,
        question: str,
        mode: OutputMode = OutputMode.CLINICAL,
        section_filter: Optional[str] = None,
        drug_filter: Optional[str] = None,
        use_hybrid: bool = True,
    ) -> RAGResponse:
        """Run the full RAG pipeline: retrieve → augment → generate."""

        # 1. Retrieve relevant label chunks
        if use_hybrid:
            hits = self._store.hybrid_search(question, k=self._k)
        else:
            hits = self._store.search(
                question, k=self._k,
                section_filter=section_filter,
                drug_filter=drug_filter,
            )

        if not hits:
            return RAGResponse(
                query=question,
                answer="No relevant drug label content found in the database. Please ingest drug labels first using scripts/ingest.py.",
                sources=[],
                mode=mode,
                model=self._model,
                tokens_used=0,
                compliance_flags=["NO_SOURCES_FOUND"],
            )

        # 2. Build the augmented prompt
        context = self._build_context(hits)
        user_prompt = self._build_user_prompt(question, context, mode)

        # 3. Generate
        completion = self._llm.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[mode]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1200,
        )

        answer = completion.choices[0].message.content
        tokens_used = completion.usage.total_tokens

        # 4. Compliance scan
        flags = self._compliance_check(answer, mode)

        return RAGResponse(
            query=question,
            answer=answer,
            sources=hits,
            mode=mode,
            model=self._model,
            tokens_used=tokens_used,
            compliance_flags=flags,
        )

    def _build_context(self, hits: list) -> str:
        parts = []
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            parts.append(
                f"[SOURCE {i}] Drug: {meta.get('drug_name', 'N/A')} | "
                f"Section: {meta.get('section_name', 'N/A')} | "
                f"Relevance: {hit['score']:.2f}\n{hit['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def _build_user_prompt(self, question: str, context: str, mode: OutputMode) -> str:
        return f"""RETRIEVED FDA LABEL CONTENT:
{context}

---

QUESTION: {question}

MODE: {mode.value.upper()}

Please answer the question using ONLY the retrieved content above.
Cite [SOURCE N] inline when drawing from a specific passage."""

    def _compliance_check(self, answer: str, mode: OutputMode) -> list:
        """Basic rule-based compliance flag detection."""
        flags = []
        answer_lower = answer.lower()

        superlatives = ["best", "safest", "superior", "only approved", "proven to cure"]
        for word in superlatives:
            if word in answer_lower:
                flags.append(f"SUPERLATIVE_CLAIM: '{word}' detected — verify label support")

        if mode == OutputMode.MARKETING:
            if not any(w in answer_lower for w in ["warning", "contraindication", "adverse", "risk"]):
                flags.append("FAIR_BALANCE_MISSING: No safety information in marketing copy")

        if "off-label" in answer_lower or "not approved" in answer_lower:
            flags.append("OFF_LABEL_REFERENCE: Manual review required")

        return flags
