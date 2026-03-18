"""
patterns/rag_fusion.py
────────────────────────────────────────────────────────────────────────────────
RAG-Fusion — full LangGraph implementation.

GRAPH STRUCTURE:
  decompose → parallel_search → rrf_merge → generate → END

KEY DESIGN: parallel search
────────────────────────────
The ideal implementation embeds and searches all N query variants in parallel
(concurrent API calls). LangGraph supports this via Send() API for true
parallelism. For simplicity and Gemini rate-limit safety, we run them
sequentially with a small delay. In production with async Gemini client,
replace the sequential loop with asyncio.gather().

WHAT THIS FIXES vs NAIVE RAG (Phase 2 failure types):
  ✓ Type B (Multi-hop): decomposition extracts both hops as separate queries
  ✓ Type C (Comparative): decomposition generates one query per side
  ✗ Type D (Negation): still single semantic direction — CRAG handles better
  ✗ Type E (Out-of-corpus): no web fallback — CRAG handles this
"""

import time
from dataclasses import dataclass, field
from typing import TypedDict
from loguru import logger

from langgraph.graph import StateGraph, END

from llm import GeminiLLM
from prompts import SYNTHESIS_PROMPT, format_context
from embedder import GeminiEmbedder
from rag_fusion_decomposer import QueryDecomposer
from rag_fusion_rrf import reciprocal_rank_fusion
from qdrant_store import QdrantStore, SearchResult


# ── State ──────────────────────────────────────────────────────────────────────

class RAGFusionState(TypedDict):
    question:        str
    query_variants:  list[str]          # N decomposed queries
    all_result_sets: list[list[SearchResult]]  # one list per variant
    merged_results:  list[SearchResult] # RRF output
    context_str:     str
    answer:          str
    latency_ms:      float


# ── Response ───────────────────────────────────────────────────────────────────

@dataclass
class RAGFusionResponse:
    question:       str
    answer:         str
    query_variants: list[str] = field(default_factory=list)
    merged_results: list[SearchResult] = field(default_factory=list)
    latency_ms:     float = 0.0

    def print_summary(self):
        print(f"\n{'='*62}")
        print(f"QUESTION: {self.question}")
        print(f"LATENCY:  {self.latency_ms:.0f}ms")
        print(f"\nQUERY VARIANTS ({len(self.query_variants)}):")
        for i, v in enumerate(self.query_variants):
            print(f"  {i+1}. {v}")
        print(f"\nRRF-MERGED TOP RESULTS ({len(self.merged_results)}):")
        for r in self.merged_results[:5]:
            print(
                f"  rrf={r.score:.5f} | "
                f"cosine={r.metadata.get('cosine_score', '?')} | "
                f"section={r.metadata.get('section','N/A')} | "
                f"pg {r.metadata.get('page','?')}"
            )
        print(f"\nANSWER:\n{self.answer}")
        print(f"{'='*62}\n")


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_rag_fusion_graph(
    embedder: GeminiEmbedder,
    store: QdrantStore,
    decomposer: QueryDecomposer,
    llm: GeminiLLM,
    top_k_per_query: int = 5,
    top_k_final: int = 5,
) -> StateGraph:

    # ── Node 1: decompose ──────────────────────────────────────────────────────
    def decompose(state: RAGFusionState) -> dict:
        """
        Generate N query variants from the original question.
        temperature=0.7 in the decomposer ensures semantic diversity.
        """
        logger.info("[RAG-Fusion] Node: decompose")
        variants = decomposer.decompose(state["question"])
        return {"query_variants": variants}

    # ── Node 2: parallel_search ────────────────────────────────────────────────
    def parallel_search(state: RAGFusionState) -> dict:
        """
        Embed and search Qdrant for each query variant.

        Returns one result set per variant. In production this would be
        async — here we run sequentially with a small rate-limit delay.

        KEY: each variant uses RETRIEVAL_QUERY task type (not DOCUMENT).
        The diversity between variants is what makes RRF effective.
        """
        logger.info(f"[RAG-Fusion] Node: parallel_search | {len(state['query_variants'])} queries")
        all_result_sets = []

        for i, variant in enumerate(state["query_variants"]):
            logger.debug(f"  Searching variant {i+1}: '{variant[:60]}...'")
            qv = embedder.embed_query(variant)
            results = store.search(query_vector=qv, top_k=top_k_per_query)
            all_result_sets.append(results)
            top_score = results[0].score if results else 0.0
            logger.debug(f"  → {len(results)} results | top score: {top_score:.3f}")

            # Small delay between embed calls for rate limiting
            if i < len(state["query_variants"]) - 1:
                time.sleep(0.2)

        total = sum(len(r) for r in all_result_sets)
        logger.info(f"[RAG-Fusion] Searched {len(all_result_sets)} variants → {total} total results")
        return {"all_result_sets": all_result_sets}

    # ── Node 3: rrf_merge ──────────────────────────────────────────────────────
    def rrf_merge(state: RAGFusionState) -> dict:
        """
        Apply Reciprocal Rank Fusion to merge all result sets.

        This is the core RAG-Fusion innovation:
          - Chunks appearing in multiple result sets get boosted scores
          - The final list is deduplicated and ranked by consensus
          - top_k_final controls how many merged results reach the generator
        """
        logger.info("[RAG-Fusion] Node: rrf_merge")
        merged = reciprocal_rank_fusion(
            result_sets=state["all_result_sets"],
            top_k=top_k_final,
        )
        context_str = format_context(merged)
        return {"merged_results": merged, "context_str": context_str}

    # ── Node 4: generate ───────────────────────────────────────────────────────
    def generate(state: RAGFusionState) -> dict:
        """
        Generate final answer from RRF-merged context.
        The context is richer and more balanced than single-query retrieval.
        """
        logger.info("[RAG-Fusion] Node: generate")
        prompt = SYNTHESIS_PROMPT.format(
            context=state["context_str"],
            question=state["question"],
        )
        answer = llm.generate(prompt, temperature=0.2)
        logger.info(f"[RAG-Fusion] Answer generated ({len(answer)} chars)")
        return {"answer": answer}

    # ── Build graph ────────────────────────────────────────────────────────────
    graph = StateGraph(RAGFusionState)
    graph.add_node("decompose",       decompose)
    graph.add_node("parallel_search", parallel_search)
    graph.add_node("rrf_merge",       rrf_merge)
    graph.add_node("generate",        generate)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose",       "parallel_search")
    graph.add_edge("parallel_search", "rrf_merge")
    graph.add_edge("rrf_merge",       "generate")
    graph.add_edge("generate",        END)

    return graph.compile()


# ── RAG-Fusion chain class ─────────────────────────────────────────────────────

class RAGFusionChain:
    """High-level interface for the RAG-Fusion pipeline."""

    def __init__(self, n_variants: int = 3, top_k: int = 5):
        logger.info("Initializing RAG-Fusion chain...")
        self.embedder   = GeminiEmbedder()
        self.store      = QdrantStore()
        self.decomposer = QueryDecomposer(n_variants=n_variants)
        self.llm        = GeminiLLM()
        self.graph      = build_rag_fusion_graph(
            embedder=self.embedder,
            store=self.store,
            decomposer=self.decomposer,
            llm=self.llm,
            top_k_per_query=top_k,
            top_k_final=top_k,
        )
        logger.success("RAG-Fusion chain ready")

    def query(self, question: str) -> RAGFusionResponse:
        start = time.time()
        initial: RAGFusionState = {
            "question":        question,
            "query_variants":  [],
            "all_result_sets": [],
            "merged_results":  [],
            "context_str":     "",
            "answer":          "",
            "latency_ms":      0.0,
        }
        final = self.graph.invoke(initial)
        latency_ms = (time.time() - start) * 1000
        return RAGFusionResponse(
            question=question,
            answer=final["answer"],
            query_variants=final["query_variants"],
            merged_results=final["merged_results"],
            latency_ms=latency_ms,
        )


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: python -m patterns.rag_fusion

    Best test cases (Phase 2 Type B and C failures):
      "What is the difference between murder and culpable homicide under IPC?"
      "Compare the RTI exemptions with Article 19 freedom of speech"
      "What happens if a person commits theft under intoxication under IPC?"
    """
    import sys
    chain = RAGFusionChain(n_variants=3, top_k=5)

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        resp = chain.query(q)
        resp.print_summary()
        sys.exit(0)

    print("\nLexRAG — RAG-Fusion Demo (Phase 5)")
    print("Type 'quit' to exit\n")

    while True:
        try:
            q = input("Your legal question: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            resp = chain.query(q)
            resp.print_summary()
        except EOFError:
            break
