"""
Vector Store & Retrieval Layer
Embeds drug label chunks and performs semantic + hybrid search.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


class EmbeddingClient:
    """Thin wrapper so you can swap embedding providers easily."""

    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Run: pip install openai")
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model

    def embed(self, texts: list) -> np.ndarray:
        """Return (N, D) float32 array of embeddings."""
        response = self._client.embeddings.create(input=texts, model=self._model)
        vecs = [d.embedding for d in response.data]
        return np.array(vecs, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class DrugLabelVectorStore:
    """
    Persisted ChromaDB collection of drug label chunks.

    Usage:
        store = DrugLabelVectorStore()
        store.add_chunks(chunks)
        results = store.search("warfarin drug interactions", k=5)
    """

    COLLECTION_NAME = "fda_drug_labels"

    def __init__(self, persist_dir: str = "./chroma_db"):
        try:
            import chromadb
        except ImportError:
            raise RuntimeError("Run: pip install chromadb")

        self._chroma = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._chroma.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = EmbeddingClient()

    def add_chunks(self, chunks, batch_size: int = 64) -> int:
        """Embed and store DrugLabelChunk objects. Returns count added."""
        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c.text for c in batch]
            ids = [f"{c.set_id}_{c.section_code}_{c.chunk_index}_{i+j}" for j, c in enumerate(batch)]
            metas = [c.to_dict() for c in batch]
            embeddings = self._embedder.embed(texts).tolist()

            self._collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metas,
            )
            added += len(batch)
            print(f"  Stored {added}/{len(chunks)} chunks...")
        return added

    def search(
        self,
        query: str,
        k: int = 5,
        section_filter: Optional[str] = None,
        drug_filter: Optional[str] = None,
    ) -> list:
        """
        Semantic similarity search with optional metadata filters.
        Returns list of dicts with keys: text, score, metadata
        """
        query_vec = self._embedder.embed_one(query).tolist()

        where = {}
        if section_filter:
            where["section_name"] = {"$eq": section_filter}
        if drug_filter:
            where["drug_name"] = {"$contains": drug_filter}

        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "score": float(1 - dist),
                "metadata": meta,
            })
        return hits

    def hybrid_search(self, query: str, k: int = 5) -> list:
        """
        Combine semantic search with simple keyword boosting.
        """
        semantic_results = self.search(query, k=k * 2)
        query_tokens = set(query.lower().split())

        for r in semantic_results:
            text_tokens = set(r["text"].lower().split())
            overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            r["score"] = 0.7 * r["score"] + 0.3 * overlap

        semantic_results.sort(key=lambda x: x["score"], reverse=True)
        return semantic_results[:k]

    def collection_stats(self) -> dict:
        count = self._collection.count()
        return {"total_chunks": count, "collection": self.COLLECTION_NAME}
