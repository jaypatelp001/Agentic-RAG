"""
patterns/crag_grader.py
────────────────────────────────────────────────────────────────────────────────
CRAG retrieval grader — scores each retrieved chunk for actual relevance.

KEY CONCEPT: Why cosine similarity alone is not enough
───────────────────────────────────────────────────────
Cosine similarity measures geometric closeness in embedding space.
This is a proxy for relevance — a good proxy, but not perfect.

Classic failure:
  Query: "Under what circumstances is a person NOT liable for murder?"
  High similarity chunk: "Section 302. Punishment for murder. Whoever commits
                          murder shall be punished with death..."

The word "murder" appears in both — high cosine similarity.
But this chunk talks about punishment, not exceptions/exemptions.
It is geometrically close but semantically WRONG for this query.

The grader catches this by asking the LLM directly:
  "Is this chunk relevant to answering the question? yes/no"

GRADING STRATEGY:
  - Grade each chunk independently (not as a batch)
  - temperature=0.0 for deterministic binary output
  - max_tokens=10 — we only need "yes" or "no"
  - If ALL chunks are irrelevant → trigger web fallback
  - If SOME chunks are relevant → filter and keep only those

PERFORMANCE COST:
  Grading adds N LLM calls per query (N = top_k = 5 by default).
  At ~200ms per grade call, this adds ~1 second to latency.
  In Phase 6 we add Redis caching to skip grading for repeated queries.
"""

from dataclasses import dataclass
from loguru import logger

from llm import GeminiLLM
from prompts import RETRIEVAL_GRADER_PROMPT
from qdrant_store import SearchResult


# ── Grade result ───────────────────────────────────────────────────────────────

@dataclass
class GradeResult:
    chunk: SearchResult
    relevant: bool          # True = keep, False = discard
    raw_grade: str          # raw LLM output ("yes" / "no" / unexpected)

    def __repr__(self):
        status = "RELEVANT" if self.relevant else "IRRELEVANT"
        section = self.chunk.metadata.get("section", "N/A")
        return f"GradeResult({status} | score={self.chunk.score:.3f} | section={section})"


# ── Grader ─────────────────────────────────────────────────────────────────────

class RetrievalGrader:
    """
    Grades retrieved chunks for actual relevance to the question.

    Decision logic:
      "yes" → relevant, keep the chunk
      "no"  → irrelevant, discard the chunk
      other → treat as irrelevant (LLM output was unexpected — be conservative)

    After grading all chunks:
      - If relevant_count == 0 → all failed → return GradeDecision.WEB_FALLBACK
      - If relevant_count > 0  → some passed → return GradeDecision.USE_FILTERED
    """

    # Decision constants — used by the CRAG chain to decide next step
    WEB_FALLBACK   = "web_fallback"
    USE_FILTERED   = "use_filtered"

    def __init__(self):
        self.llm = GeminiLLM()

    def grade_chunk(self, question: str, chunk: SearchResult) -> GradeResult:
        """Grade a single chunk against the question."""
        prompt = RETRIEVAL_GRADER_PROMPT.format(
            question=question,
            document=chunk.text[:800],  # truncate very long chunks for grading
        )
        raw = self.llm.grade(prompt)  # temperature=0.0, max_tokens=10
        relevant = raw.startswith("yes")

        logger.debug(
            f"  Grade: {'YES' if relevant else 'NO '} | "
            f"score={chunk.score:.3f} | "
            f"section={chunk.metadata.get('section', 'N/A')} | "
            f"raw='{raw}'"
        )
        return GradeResult(chunk=chunk, relevant=relevant, raw_grade=raw)

    def grade_all(
        self,
        question: str,
        chunks: list[SearchResult],
    ) -> tuple[str, list[SearchResult]]:
        """
        Grade all retrieved chunks and decide the next action.

        Returns:
            (decision, filtered_chunks)
            decision = WEB_FALLBACK | USE_FILTERED
            filtered_chunks = only the relevant chunks (empty if WEB_FALLBACK)
        """
        logger.info(f"Grading {len(chunks)} chunks for relevance...")
        grades = [self.grade_chunk(question, chunk) for chunk in chunks]

        relevant_chunks = [g.chunk for g in grades if g.relevant]
        irrelevant_count = len(grades) - len(relevant_chunks)

        logger.info(
            f"Grading complete: {len(relevant_chunks)} relevant, "
            f"{irrelevant_count} irrelevant out of {len(chunks)}"
        )

        if not relevant_chunks:
            logger.warning("All chunks graded irrelevant → triggering web fallback")
            return self.WEB_FALLBACK, []

        return self.USE_FILTERED, relevant_chunks
