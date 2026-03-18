"""
patterns/rag_fusion_rrf.py
────────────────────────────────────────────────────────────────────────────────
Reciprocal Rank Fusion (RRF) — merge and re-rank multiple result sets.

KEY CONCEPT: What RRF does and why it works
────────────────────────────────────────────
Given N result sets (one per query variant), RRF assigns each document
a combined score based on its rank position across all sets.

Formula:  RRF_score(doc) = Σ  1 / (k + rank_i)
                          i=1..N

where:
  k    = 60  (constant — dampens the impact of top-1 dominance)
  rank = position in result set i (1-indexed)
  N    = number of result sets

WHY k=60?
  Without k, rank-1 in a single set scores 1.0 and rank-2 scores 0.5 —
  too much weight on single-set winners.
  k=60 makes rank-1 score 1/61≈0.016 and rank-2 score 1/62≈0.016 —
  much flatter, so a doc appearing at rank 3 in all 3 sets beats
  a doc at rank 1 in only one set.

  k=60 is the value from the original RRF paper (Cormack et al., 2009)
  and remains the standard default.

DEDUPLICATION:
  The same chunk can appear in multiple result sets (retrieved by different
  query variants). RRF naturally handles this: each appearance adds to the
  chunk's cumulative score. A chunk retrieved by all 3 queries gets 3 terms
  in its sum — this is exactly what we want.
  We deduplicate by chunk_id before returning the final list.

RESULT:
  Returns a single ranked list of SearchResult objects.
  The .score field is replaced with the RRF score for transparency.
  Typically top_k=5 from the merged list is passed to the generator.
"""

from dataclasses import dataclass, field
from loguru import logger
from qdrant_store import SearchResult


RRF_K = 60   # standard constant from Cormack et al. 2009


def reciprocal_rank_fusion(
    result_sets: list[list[SearchResult]],
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Merge N result sets using Reciprocal Rank Fusion.

    Args:
        result_sets: List of result lists, one per query variant.
                     Each inner list is sorted by descending similarity score.
        top_k:       Number of results to return after merging.

    Returns:
        Merged and re-ranked list of SearchResult, sorted by RRF score desc.
        The .score field contains the RRF score (not cosine similarity).
    """
    # Accumulate RRF scores keyed by chunk_id
    rrf_scores: dict[str, float] = {}
    # Store the SearchResult object for each chunk_id (keep highest-score version)
    chunk_store: dict[str, SearchResult] = {}

    for result_list in result_sets:
        for rank, result in enumerate(result_list, start=1):
            chunk_id = result.metadata.get("chunk_id", f"unknown_{rank}")

            # RRF formula: add 1/(k + rank) to cumulative score
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

            # Store result — prefer the one with higher original cosine score
            if chunk_id not in chunk_store or result.score > chunk_store[chunk_id].score:
                chunk_store[chunk_id] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    # Build final list — replace .score with RRF score for transparency
    merged = []
    for chunk_id in sorted_ids[:top_k]:
        result = chunk_store[chunk_id]
        # Annotate metadata with RRF score and original cosine score
        enriched_meta = {
            **result.metadata,
            "rrf_score":   round(rrf_scores[chunk_id], 6),
            "cosine_score": round(result.score, 4),
        }
        merged.append(SearchResult(
            text=result.text,
            score=rrf_scores[chunk_id],   # use RRF score as .score
            metadata=enriched_meta,
        ))

    logger.info(
        f"RRF merged {sum(len(r) for r in result_sets)} results from "
        f"{len(result_sets)} sets → {len(merged)} unique chunks"
    )
    for i, r in enumerate(merged[:3]):
        logger.debug(
            f"  [{i+1}] rrf={r.score:.5f} | "
            f"cosine={r.metadata['cosine_score']:.3f} | "
            f"section={r.metadata.get('section','N/A')}"
        )

    return merged


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: python -m patterns.rag_fusion_rrf

    Demonstrates why RRF promotes docs appearing in multiple sets.
    Chunk C appears in all 3 sets (ranks 3, 1, 2) → gets highest RRF score.
    Chunk A appears only in set 1 at rank 1 → gets lower RRF score than C.
    """
    def make_result(chunk_id, section, score):
        return SearchResult(
            text=f"Text for {chunk_id}",
            score=score,
            metadata={"chunk_id": chunk_id, "section": section, "source": "test.pdf", "page": 1}
        )

    set1 = [
        make_result("A", "Section 302", 0.91),
        make_result("B", "Section 300", 0.85),
        make_result("C", "Section 299", 0.79),
    ]
    set2 = [
        make_result("C", "Section 299", 0.88),
        make_result("D", "Section 304", 0.81),
        make_result("A", "Section 302", 0.75),
    ]
    set3 = [
        make_result("B", "Section 300", 0.90),
        make_result("C", "Section 299", 0.84),
        make_result("E", "Section 303", 0.77),
    ]

    merged = reciprocal_rank_fusion([set1, set2, set3], top_k=5)

    print("\nRRF Results (should rank C first — appears in all 3 sets):")
    for i, r in enumerate(merged):
        print(
            f"  [{i+1}] chunk={r.metadata['chunk_id']} | "
            f"section={r.metadata['section']} | "
            f"rrf={r.score:.5f} | cosine={r.metadata['cosine_score']:.3f}"
        )

    print(f"\nExpected: C first (rrf≈{1/(60+3)+1/(60+1)+1/(60+2):.5f})")
    print(f"  A: rrf≈{1/(60+1)+1/(60+3):.5f}  (rank 1 in set1, rank 3 in set2)")
    print(f"  C: rrf≈{1/(60+3)+1/(60+1)+1/(60+2):.5f}  (rank 3,1,2 across all sets)")
