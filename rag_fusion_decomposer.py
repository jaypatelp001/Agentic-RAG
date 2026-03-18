"""
patterns/rag_fusion_decomposer.py
────────────────────────────────────────────────────────────────────────────────
Query decomposition for RAG-Fusion — rewrites one query into N variants.

KEY CONCEPT: Why single-query retrieval fails on complex questions
──────────────────────────────────────────────────────────────────
Consider: "What is the difference between murder and culpable homicide under IPC?"

Single embedding approach:
  The query embeds to a vector somewhere BETWEEN "murder" and "culpable homicide"
  in the 768-dim space. It's close to neither concept specifically.
  Qdrant returns chunks that mention both terms — often procedural text that
  uses both words but explains neither clearly.

RAG-Fusion approach — decompose into 3 targeted queries:
  Q1: "IPC Section 302 murder definition and punishment"
  Q2: "IPC Section 299 300 culpable homicide definition elements"
  Q3: "difference between murder and culpable homicide mens rea intention"

  Each query retrieves the precise chunks for its side of the comparison.
  RRF merges the 3 result sets into one ranked list.
  The final context has both sides explained cleanly → Gemini can compare.

WHAT MAKES A GOOD DECOMPOSITION:
  1. Each variant covers a different angle (not just paraphrases)
  2. Variants use specific legal terminology when possible
  3. One variant should always be close to the original (don't over-decompose)
  4. temperature=0.7 for variety — we WANT different embedding directions

HOW MANY VARIANTS:
  2 → barely better than single query (too similar)
  3 → sweet spot for legal domain (our default)
  5 → diminishing returns, higher latency, more noise in merged results
"""

import re
import json
from loguru import logger
from llm import GeminiLLM
from prompts import QUERY_DECOMPOSE_PROMPT


class QueryDecomposer:
    """
    Decomposes a complex legal query into N targeted sub-queries.

    Each sub-query is designed to retrieve a different facet of the answer:
      - Different legal concepts (both sides of a comparison)
      - Different sections/acts (cross-document questions)
      - Different specificity levels (broad + specific)
    """

    def __init__(self, n_variants: int = 3):
        self.n_variants = n_variants
        self.llm = GeminiLLM()

    def decompose(self, question: str) -> list[str]:
        """
        Generate N query variants for a given question.

        Returns a list of strings. Always includes the original question
        as the first variant (fallback if decomposition fails).

        Parsing strategy:
          Gemini returns a Python list literal: ["q1", "q2", "q3"]
          We parse with ast.literal_eval for safety (no exec/eval).
          On parse failure, return [original_question] × n_variants.
        """
        logger.info(f"Decomposing query into {self.n_variants} variants...")
        logger.debug(f"  Original: '{question[:60]}...'")

        prompt = QUERY_DECOMPOSE_PROMPT.format(
            question=question,
            n_variants=self.n_variants,
        )

        raw = self.llm.decompose(prompt)   # temperature=0.7

        variants = self._parse_variants(raw, question)
        logger.info(f"Decomposed into {len(variants)} variants:")
        for i, v in enumerate(variants):
            logger.debug(f"  Q{i+1}: '{v[:70]}'")

        return variants

    def _parse_variants(self, raw: str, fallback: str) -> list[str]:
        """
        Parse Gemini's list output safely.

        Handles common output formats:
          ["query 1", "query 2", "query 3"]        → standard
          ["query 1",\n "query 2",\n "query 3"]    → multiline
          1. query 1\n2. query 2\n3. query 3        → numbered list fallback
        """
        import ast

        # Try JSON array first
        try:
            bracket_match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if bracket_match:
                variants = ast.literal_eval(bracket_match.group())
                if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
                    # Always include original as anchor query
                    if fallback not in variants:
                        variants.insert(0, fallback)
                    return variants[:self.n_variants + 1]
        except Exception:
            pass

        # Try numbered list format: "1. query\n2. query"
        numbered = re.findall(r'\d+\.\s+(.+?)(?=\n\d+\.|\Z)', raw, re.DOTALL)
        if len(numbered) >= 2:
            variants = [v.strip().strip('"') for v in numbered]
            if fallback not in variants:
                variants.insert(0, fallback)
            return variants[:self.n_variants + 1]

        # Fallback: return original query repeated (safe but not ideal)
        logger.warning(f"Could not parse decomposition output — using original query only")
        return [fallback] * self.n_variants


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: python -m patterns.rag_fusion_decomposer

    Expected — 3 variants covering different angles:
      Q1: Close to original (anchor)
      Q2: One specific concept
      Q3: The other concept or the comparison angle
    """
    decomposer = QueryDecomposer(n_variants=3)

    test_questions = [
        "What is the difference between murder and culpable homicide under IPC?",
        "How do RTI exemptions compare to constitutional freedom of speech?",
        "What happens if a PIO fails to respond to RTI and the applicant files a second appeal?",
    ]

    for q in test_questions:
        print(f"\nOriginal: {q}")
        print("Variants:")
        variants = decomposer.decompose(q)
        for i, v in enumerate(variants):
            print(f"  {i+1}. {v}")
