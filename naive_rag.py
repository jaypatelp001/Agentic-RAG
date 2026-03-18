"""
patterns/naive_rag.py
────────────────────────────────────────────────────────────────────────────────
Phase 2: Baseline naive RAG — the simplest possible implementation.

PURPOSE: Build it clean, then deliberately break it.
─────────────────────────────────────────────────────
This intentionally has NO:
  - Retrieval grading       (added in Phase 3: CRAG)
  - Web fallback            (added in Phase 3: CRAG)
  - Reflection / self-check (added in Phase 4: Self-RAG)
  - Multi-query retrieval   (added in Phase 5: RAG-Fusion)
  - SQL routing             (added in Phase 5: Adaptive)

This minimal chain is your baseline. Run failure_analysis.py after building
this to see exactly where and why it breaks — that experience motivates
every advanced pattern you'll implement next.

WHAT YOU LEARN HERE:
  1. The complete RAG request/response cycle
  2. What "context window stuffing" looks like in practice
  3. How retrieval score distribution tells you about query quality
  4. The specific failure modes of single-shot retrieval on legal text
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from llm import GeminiLLM
from prompts import NAIVE_RAG_PROMPT, format_context
from embedder import GeminiEmbedder
from qdrant_store import QdrantStore, SearchResult


# ── Response dataclass ─────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Complete response from the RAG chain — answer + full audit trail.

    The audit trail (retrieved_chunks, scores, latency) is critical for
    failure analysis. Don't throw it away — it's how you diagnose bad answers.
    """
    question: str
    answer: str
    retrieved_chunks: list[SearchResult] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def top_score(self) -> float:
        """Highest relevance score among retrieved chunks."""
        return max((r.score for r in self.retrieved_chunks), default=0.0)

    @property
    def avg_score(self) -> float:
        """Average relevance score — low avg = poor retrieval overall."""
        if not self.retrieved_chunks:
            return 0.0
        return sum(r.score for r in self.retrieved_chunks) / len(self.retrieved_chunks)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"QUESTION: {self.question}")
        print(f"{'='*60}")
        print(f"\nANSWER:\n{self.answer}")
        print(f"\nRETRIEVAL STATS:")
        print(f"  Chunks retrieved: {len(self.retrieved_chunks)}")
        print(f"  Top score:        {self.top_score:.3f}")
        print(f"  Avg score:        {self.avg_score:.3f}")
        print(f"  Latency:          {self.latency_ms:.0f}ms")
        print(f"\nSOURCES RETRIEVED:")
        for i, chunk in enumerate(self.retrieved_chunks, 1):
            print(f"  [{i}] score={chunk.score:.3f} | "
                  f"{chunk.metadata.get('act_name','?')} | "
                  f"page={chunk.metadata.get('page','?')} | "
                  f"section={chunk.metadata.get('section','N/A')}")
        print(f"{'='*60}\n")


# ── Naive RAG chain ────────────────────────────────────────────────────────────

class NaiveRAG:
    """
    Minimal RAG: embed query → vector search → stuff context → generate.

    This is a straight pipeline — no feedback, no correction, no loops.
    It works well for simple queries but fails predictably on:
      - Multi-hop questions ("Under IPC, what is the difference between...")
      - Ambiguous queries (retrieves wrong sections)
      - Questions outside the corpus (hallucinates instead of saying "I don't know")
      - Specific section lookups where the chunk boundary cuts mid-section

    Run failure_analysis.py to see all these failure modes quantified.
    """

    def __init__(
        self,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        """
        top_k: number of chunks to retrieve per query.
               Too low (1-2): not enough context.
               Too high (10+): irrelevant chunks dilute the signal.
               5 is a good baseline for legal text.

        score_threshold: minimum cosine similarity to include a chunk.
               0.0 = include everything (naive baseline — no filtering).
               CRAG will add proper grading on top of this.
        """
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.embedder = GeminiEmbedder()
        self.store = QdrantStore()
        self.llm = GeminiLLM()
        logger.info(f"NaiveRAG ready | top_k={top_k} | score_threshold={score_threshold}")

    def query(self, question: str) -> RAGResponse:
        """
        Full naive RAG pipeline for a single question.

        Steps:
          1. Embed the query (RETRIEVAL_QUERY task type)
          2. Search Qdrant for top-k similar chunks
          3. Format chunks as numbered context string
          4. Build prompt = system instructions + context + question
          5. Call Gemini to generate the answer
          6. Return RAGResponse with full audit trail
        """
        start = time.time()

        # ── Step 1: Embed query ────────────────────────────────────────────────
        logger.debug(f"Embedding query: '{question[:60]}...'")
        query_vector = self.embedder.embed_query(question)

        # ── Step 2: Retrieve chunks ────────────────────────────────────────────
        logger.debug(f"Searching Qdrant (top_k={self.top_k})...")
        results = self.store.search(
            query_vector=query_vector,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )
        logger.debug(f"Retrieved {len(results)} chunks | scores: {[round(r.score,3) for r in results]}")

        # ── Step 3: Format context ─────────────────────────────────────────────
        context = format_context(results)

        # ── Step 4 + 5: Generate answer ────────────────────────────────────────
        prompt = NAIVE_RAG_PROMPT.format(
            context=context,
            question=question,
        )
        logger.debug("Calling Gemini for generation...")
        answer = self.llm.generate(prompt, temperature=0.2)

        latency_ms = (time.time() - start) * 1000

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=results,
            latency_ms=latency_ms,
            metadata={
                "pattern": "naive_rag",
                "top_k": self.top_k,
                "score_threshold": self.score_threshold,
            }
        )


# ── Interactive demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: python -m patterns.naive_rag

    Try asking:
      - "What is the punishment for murder under IPC?"        (should work well)
      - "What are the grounds for bail under CrPC?"           (moderate)
      - "Compare RTI exemptions with Constitution Article 19" (will likely fail)
    """
    rag = NaiveRAG(top_k=5)

    print("\nLexRAG — Naive RAG Demo (Phase 2)")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Your legal question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        response = rag.query(question)
        response.print_summary()