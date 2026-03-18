"""
ingestion/embedder.py
────────────────────────────────────────────────────────────────────────────────
Chunks → float vectors using Gemini text-embedding-004.

KEY CONCEPT: What embeddings are and why they work
───────────────────────────────────────────────────
text-embedding-004 maps text → 768-dimensional float vector.

"Section 302 murder punishment" → [0.023, -0.41, 0.87, ...]  (768 numbers)
"What is the penalty for killing?" → [0.031, -0.39, 0.84, ...]  (768 numbers)

These two vectors are geometrically close (high cosine similarity).
Qdrant finds the nearest vectors to your query vector — that's retrieval.

WHY TASK TYPE MATTERS:
Gemini's embedding API accepts a `task_type` parameter.
  - RETRIEVAL_DOCUMENT: used when embedding chunks for storage
  - RETRIEVAL_QUERY:    used when embedding user queries at search time
Using the wrong task_type silently degrades retrieval quality by 10-20%.
This is one of the most common production RAG bugs.

BATCHING:
The Gemini API has rate limits. We batch chunks in groups of 100
and add retry logic with exponential backoff. In production you'd
also add Redis caching here to avoid re-embedding identical chunks.
"""

import os
import time
from typing import Literal
from loguru import logger

import google.generativeai as genai
from dotenv import load_dotenv

from chunker import Chunk

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM   = 3072      # gemini-embedding-001 output dimension
BATCH_SIZE      = 100       # Gemini API: up to 100 texts per batch call
MAX_RETRIES     = 3
RETRY_DELAY     = 2.0       # seconds, doubles on each retry


# ── Embedder class ─────────────────────────────────────────────────────────────

class GeminiEmbedder:
    """
    Wraps the Gemini embedding API with batching, retry, and task_type handling.

    Usage:
        embedder = GeminiEmbedder()
        vectors = embedder.embed_chunks(chunks)
        # vectors[i] is the 768-dim embedding for chunks[i]
    """

    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set. Check your .env file.")
        genai.configure(api_key=key)
        logger.info(f"GeminiEmbedder ready — model: {EMBEDDING_MODEL}, dim: {EMBEDDING_DIM}")

    def _embed_batch(
        self,
        texts: list[str],
        task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"] = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """
        Embed a batch of texts with retry logic.

        task_type:
          RETRIEVAL_DOCUMENT → use when indexing chunks into Qdrant
          RETRIEVAL_QUERY    → use when embedding user queries at search time
        """
        for attempt in range(MAX_RETRIES):
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=texts,
                    task_type=task_type,
                )
                return result["embedding"] if len(texts) == 1 else result["embedding"]
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} attempts: {e}")
                wait = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Embed attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """
        Embed all chunks for storage (uses RETRIEVAL_DOCUMENT task type).
        Returns a list of 768-dim vectors, one per chunk, in the same order.
        """
        texts = [chunk.text for chunk in chunks]
        all_vectors = []
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info(f"Embedding {len(chunks)} chunks in {total_batches} batches...")

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            vectors = self._embed_batch(batch, task_type="RETRIEVAL_DOCUMENT")
            all_vectors.extend(vectors)

            logger.debug(f"  Batch {batch_num}/{total_batches} done ({len(batch)} chunks)")

            # Rate limiting: small pause between batches
            if batch_num < total_batches:
                time.sleep(0.1)

        logger.success(f"Embedded {len(all_vectors)} chunks → each is {EMBEDDING_DIM}-dim vector")
        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single user query for retrieval (uses RETRIEVAL_QUERY task type).

        IMPORTANT: This MUST use RETRIEVAL_QUERY, not RETRIEVAL_DOCUMENT.
        Gemini's embedding model is trained differently for queries vs documents.
        Using RETRIEVAL_DOCUMENT for queries degrades retrieval accuracy.
        """
        vectors = self._embed_batch([query], task_type="RETRIEVAL_QUERY")
        # For a single text, the API returns a list with one vector
        return vectors[0] if isinstance(vectors[0], list) else vectors


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: python -m ingestion.embedder
    Expected output:
      GeminiEmbedder ready — model: models/text-embedding-004, dim: 768
      Embedding 2 chunks in 1 batches...
      Embedded 2 chunks → each is 768-dim vector
      Vector 0 shape: 768 floats, range [-0.12, 0.08]
      Query vector:   768 floats

      SIMILARITY TEST (cosine):
        query: 'What is the punishment for murder?'
        chunk 0 similarity: 0.847  ← Section 302 (murder) — high similarity
        chunk 1 similarity: 0.312  ← Section 420 (fraud) — low similarity
    """
    import numpy as np
    from chunker import Chunk

    sample_chunks = [
        Chunk(
            text="Section 302. Punishment for murder. Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
            metadata={"chunk_id": "test_c0", "source": "ipc.pdf", "section": "Section 302"}
        ),
        Chunk(
            text="Section 420. Cheating and dishonestly inducing delivery of property. Whoever cheats and thereby dishonestly induces the person deceived to deliver any property shall be punished.",
            metadata={"chunk_id": "test_c1", "source": "ipc.pdf", "section": "Section 420"}
        ),
    ]

    embedder = GeminiEmbedder()
    vectors = embedder.embed_chunks(sample_chunks)
    query_vec = embedder.embed_query("What is the punishment for murder?")

    print(f"\nVector 0 shape: {len(vectors[0])} floats, range [{min(vectors[0]):.2f}, {max(vectors[0]):.2f}]")
    print(f"Query vector:   {len(query_vec)} floats")

    # Cosine similarity
    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    print(f"\nSIMILARITY TEST (cosine):")
    print(f"  query: 'What is the punishment for murder?'")
    for i, (chunk, vec) in enumerate(zip(sample_chunks, vectors)):
        sim = cosine(query_vec, vec)
        print(f"  chunk {i} similarity: {sim:.3f}  ← {chunk.metadata['section']}")