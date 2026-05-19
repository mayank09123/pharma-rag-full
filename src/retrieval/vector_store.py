from __future__ import annotations
import os
import numpy as np
from typing import Optional


# =========================
# Embeddings (OpenAI SAFE)
# =========================
class EmbeddingClient:
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing OPENAI_API_KEY (set in Streamlit Secrets)")

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list) -> np.ndarray:
        response = self._client.embeddings.create(
            input=texts,
            model=self._model
        )
        return np.array([d.embedding for d in response.data], dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# =========================
# Cohere Reranker (SAFE)
# =========================
class CohereReranker:
    MODEL = "rerank-english-v3.0"

    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")

        if not self.api_key:
            print("⚠ Cohere rerank disabled (no API key)")
            self._co = None
            return

        try:
            import cohere
            self._co = cohere.Client(self.api_key)
        except Exception as e:
            print(f"⚠ Cohere init failed: {e}")
            self._co = None

    def rerank(self, query: str, chunks: list, top_n: int = 5) -> list:
        if not self._co or not chunks:
            return chunks[:top_n]

        documents = [c["text"] for c in chunks]

        try:
            response = self._co.rerank(
                query=query,
                documents=documents,
                top_n=min(top_n, len(chunks)),
                model=self.MODEL,
                return_documents=False,
            )
        except Exception as e:
            print(f"⚠ Rerank API failed: {e}")
            return chunks[:top_n]

        reranked = []
        for r in response.results:
            chunk = chunks[r.index].copy()
            chunk["original_score"] = chunk["score"]
            chunk["score"] = r.relevance_score
            chunk["reranked"] = True
            reranked.append(chunk)

        print(f"✓ Cohere Rerank: {len(chunks)} → {len(reranked)}")
        return reranked


# =========================
# Vector Store (ChromaDB)
# =========================
class DrugLabelVectorStore:

    COLLECTION_NAME = "fda_drug_labels"

    def __init__(self,
                 persist_dir: str = None,
                 use_rerank: bool = True):

        import chromadb

        # SAFE path (important for Streamlit Cloud)
        if persist_dir is None:
            persist_dir = os.getenv("CHROMA_PATH", "./chroma_db")

        self._chroma = chromadb.PersistentClient(path=persist_dir)

        self._collection = self._chroma.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self._embedder = EmbeddingClient()
        self._use_rerank = use_rerank

        # Safe reranker init
        self._reranker = CohereReranker()
        if not self._reranker._co:
            self._use_rerank = False

    # =========================
    # Add chunks
    # =========================
    def add_chunks(self, chunks, batch_size: int = 64) -> int:
        added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            texts = [c.text for c in batch]
            ids = [
                f"{c.set_id}_{c.section_code}_{c.chunk_index}_{i+j}"
                for j, c in enumerate(batch)
            ]
            metas = [c.to_dict() for c in batch]
            embeddings = self._embedder.embed(texts).tolist()

            self._collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metas,
            )

            added += len(batch)
            print(f"Stored {added}/{len(chunks)} chunks...")

        return added

    # =========================
    # Basic search
    # =========================
    def search(self, query: str, k: int = 5,
               section_filter: Optional[str] = None,
               drug_filter: Optional[str] = None):

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

        return [
            {
                "text": doc,
                "score": float(1 - dist),
                "metadata": meta,
                "reranked": False
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # =========================
    # Hybrid + rerank search
    # =========================
    def hybrid_search(self, query: str, k: int = 5):

        candidate_k = k * 4

        results = self.search(query, k=candidate_k)

        # simple keyword boost
        query_tokens = set(query.lower().split())

        for r in results:
            overlap = len(
                query_tokens & set(r["text"].lower().split())
            ) / max(len(query_tokens), 1)

            r["score"] = 0.7 * r["score"] + 0.3 * overlap

        results.sort(key=lambda x: x["score"], reverse=True)

        # rerank safely
        if self._use_rerank and self._reranker:
            results = self._reranker.rerank(query, results, top_n=k)
        else:
            results = results[:k]

        return results

    # =========================
    # Stats
    # =========================
    def collection_stats(self):
        return {
            "total_chunks": self._collection.count(),
            "collection": self.COLLECTION_NAME,
            "rerank_enabled": self._use_rerank,
        }