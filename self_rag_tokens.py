"""
patterns/self_rag_tokens.py
────────────────────────────────────────────────────────────────────────────────
Self-RAG reflection tokens — 4 binary LLM checkpoints.

KEY CONCEPT: What reflection tokens are
─────────────────────────────────────────
The original Self-RAG paper (Asai et al., 2023) trains a special model
that produces reflection tokens as part of its output vocabulary.
We implement the same LOGIC using separate grading LLM calls.

The 4 tokens and what they catch:

  IsRETRIEVE — "Should I even retrieve documents for this question?"
    yes → proceed to Qdrant search
    no  → question is common knowledge, answer directly (e.g. "What year was
           the IPC enacted?" — don't waste a retrieval call)

  IsREL — "Are the retrieved chunks actually relevant?"
    yes → proceed to generation with this context
    no  → chunks are off-topic, generate from parametric memory
           (IMPORTANT: this is different from CRAG's web fallback —
            Self-RAG trusts the LLM's own knowledge as fallback,
            not an external source)

  IsSUP — "Is the generated answer fully supported by the context?"
    yes → answer is grounded, proceed to IsUSE check
    no  → answer contains claims not in the context (hallucination risk)
           → regenerate with a stricter prompt

  IsUSE — "Is the generated answer actually useful for the question?"
    yes → return the answer
    no  → answer is technically correct but doesn't address what was asked
           → regenerate with more specific instructions

WHY ALL 4 MATTER:
  IsSUP catches hallucination (content error)
  IsUSE catches relevance failures (the answer is true but unhelpful)
  These two failures are orthogonal — you need both checks.

IMPLEMENTATION NOTE:
  All tokens use temperature=0.0 (deterministic).
  We parse for "yes"/"no" prefix — unexpected outputs default to "no"
  (conservative — better to regenerate than to return a bad answer).
"""

from loguru import logger
from llm import GeminiLLM
from prompts import (
    SELF_RAG_RETRIEVE_PROMPT,
    SELF_RAG_RELEVANCE_PROMPT,
    SELF_RAG_SUPPORT_PROMPT,
    SELF_RAG_USEFUL_PROMPT,
)
from qdrant_store import SearchResult


class ReflectionTokens:
    """
    Evaluates all 4 Self-RAG reflection tokens.
    Each token is a binary yes/no LLM call at temp=0.0.
    """

    MAX_REGENERATE = 2   # max re-generation attempts before giving up

    def __init__(self):
        self.llm = GeminiLLM()

    def _binary(self, prompt: str, token_name: str) -> bool:
        """
        Make a binary yes/no decision using the grading LLM.
        Returns True for 'yes', False for anything else.
        Conservative default: unexpected output → False (re-evaluate / regenerate).
        """
        raw = self.llm.grade(prompt)   # temp=0.0, max_tokens=10
        result = raw.startswith("yes")
        logger.debug(f"  [{token_name}] raw='{raw}' → {'YES' if result else 'NO'}")
        return result

    # ── Token 1: IsRETRIEVE ───────────────────────────────────────────────────

    def is_retrieve(self, question: str) -> bool:
        """
        Should we retrieve documents for this question?

        Returns False (skip retrieval) for:
          - Simple definitional questions the LLM knows cold
          - Questions about well-known historical facts
          - Questions where the answer is a single well-known number/date

        Returns True (do retrieval) for:
          - Specific section numbers, legal provisions
          - Procedural questions
          - Anything that needs citation

        In practice for legal RAG: almost always True.
        The token's value becomes more apparent in general-domain RAG.
        """
        prompt = SELF_RAG_RETRIEVE_PROMPT.format(question=question)
        result = self._binary(prompt, "IsRETRIEVE")
        logger.info(f"[Self-RAG] IsRETRIEVE: {'RETRIEVE' if result else 'SKIP — answer directly'}")
        return result

    # ── Token 2: IsREL ────────────────────────────────────────────────────────

    def is_relevant(self, question: str, chunk: SearchResult) -> bool:
        """
        Are the retrieved chunks relevant to the question?
        Evaluated per-chunk (same as CRAG grader).

        KEY DIFFERENCE from CRAG grader:
          CRAG: irrelevant → web search fallback
          Self-RAG: irrelevant → generate from parametric memory (no web search)

        This makes Self-RAG more self-contained but potentially less accurate
        on out-of-corpus questions. CRAG's web fallback is more robust for
        legal use cases where the corpus may be incomplete.
        """
        prompt = SELF_RAG_RELEVANCE_PROMPT.format(
            question=question,
            document=chunk.text[:600],
        )
        return self._binary(prompt, "IsREL")

    def filter_relevant(self, question: str, chunks: list[SearchResult]) -> list[SearchResult]:
        """Grade all chunks and return only the relevant ones."""
        logger.info(f"[Self-RAG] IsREL — grading {len(chunks)} chunks...")
        relevant = [c for c in chunks if self.is_relevant(question, c)]
        logger.info(f"[Self-RAG] IsREL: {len(relevant)}/{len(chunks)} chunks relevant")
        return relevant

    # ── Token 3: IsSUP ────────────────────────────────────────────────────────

    def is_supported(self, context: str, answer: str) -> bool:
        """
        Is every claim in the generated answer supported by the context?

        This is the hallucination detector. It checks whether the LLM
        introduced facts that don't appear in the retrieved chunks.

        Example failure caught by IsSUP:
          Context:  "Section 302 — punishment is death or life imprisonment"
          Answer:   "Section 302 mandates death penalty and a minimum fine of
                     Rs. 10,000" ← the fine amount is NOT in the context

        IsSUP catches this and triggers regeneration with a stricter prompt.
        """
        prompt = SELF_RAG_SUPPORT_PROMPT.format(
            context=context[:2000],
            answer=answer[:1000],
        )
        result = self._binary(prompt, "IsSUP")
        logger.info(f"[Self-RAG] IsSUP: {'SUPPORTED' if result else 'UNSUPPORTED — regenerating'}")
        return result

    # ── Token 4: IsUSE ────────────────────────────────────────────────────────

    def is_useful(self, question: str, answer: str) -> bool:
        """
        Is the answer actually useful for what was asked?

        Catches a different failure than IsSUP:
          IsSUP: answer is hallucinated (false claims)
          IsUSE: answer is factually grounded but doesn't address the question

        Example failure caught by IsUSE:
          Question: "What is the procedure for RTI appeal?"
          Answer:   "The RTI Act was enacted in 2005 to promote transparency
                     in government. It covers all public authorities..."
          → Factually true (IsSUP passes) but doesn't answer the procedure question.
          → IsUSE fails → regenerate with more specific instructions.
        """
        prompt = SELF_RAG_USEFUL_PROMPT.format(
            question=question,
            answer=answer[:1000],
        )
        result = self._binary(prompt, "IsUSE")
        logger.info(f"[Self-RAG] IsUSE: {'USEFUL' if result else 'NOT USEFUL — regenerating'}")
        return result

    # ── Combined post-generation check ────────────────────────────────────────

    def passes_post_checks(self, question: str, context: str, answer: str) -> tuple[bool, str]:
        """
        Run IsSUP and IsUSE together after generation.

        Returns:
            (passes: bool, reason: str)
            passes=True  → answer is good, return it
            passes=False → answer failed a check, caller should regenerate
        """
        if not self.is_supported(context, answer):
            return False, "IsSUP_FAILED: answer contains unsupported claims"
        if not self.is_useful(question, answer):
            return False, "IsUSE_FAILED: answer does not address the question"
        return True, "OK"
