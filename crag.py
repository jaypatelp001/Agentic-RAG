"""
patterns/crag.py
────────────────────────────────────────────────────────────────────────────────
Corrective RAG (CRAG) — full LangGraph implementation.

KEY CONCEPT: LangGraph state machines for RAG
──────────────────────────────────────────────
LangGraph models the RAG pipeline as a graph:
  - Nodes = processing steps (retrieve, grade, web_search, generate)
  - Edges = flow between steps
  - Conditional edges = branching based on state

State flows through the graph, accumulating results at each node.
Conditional edges allow CRAG's "if bad → fallback" logic to be expressed
cleanly as routing functions rather than nested if/else blocks.

CRAG GRAPH STRUCTURE:
  retrieve → grade → [USE_FILTERED] → generate
                   → [WEB_FALLBACK] → web_search → generate

WHAT CRAG FIXES vs NAIVE RAG (from your Phase 2 failure analysis):
  ✓ Type D (Negation):     Grader rejects wrong chunks, web search finds exceptions
  ✓ Type E (Out-of-corpus): Grader rejects all chunks → web fallback → real answer
  ✗ Type B (Multi-hop):    Still single query — RAG-Fusion fixes this (Phase 5)
  ✗ Type C (Comparative):  Still single query — RAG-Fusion fixes this (Phase 5)
"""

import time
from dataclasses import dataclass, field
from typing import Literal, TypedDict
from loguru import logger

from langgraph.graph import StateGraph, END

from llm import GeminiLLM
from prompts import SYNTHESIS_PROMPT, format_context
from embedder import GeminiEmbedder
from crag_grader import RetrievalGrader
from qdrant_store import QdrantStore, SearchResult
from web_search import TavilyWebSearch, WebResult


# ── Graph state ────────────────────────────────────────────────────────────────

class CRAGState(TypedDict):
    """
    The state object that flows through every node in the LangGraph.

    KEY CONCEPT: Why TypedDict for state?
    ──────────────────────────────────────
    LangGraph requires the state to be a TypedDict (or Pydantic model).
    Every node receives the full state and returns a partial update.
    LangGraph merges the updates — nodes don't overwrite each other's fields.

    Think of it as a shared blackboard that every node can read and write to.
    """
    question:         str                    # original user question
    query_vector:     list[float]            # embedded query vector
    retrieved_chunks: list[SearchResult]     # raw Qdrant results
    grade_decision:   str                    # WEB_FALLBACK | USE_FILTERED
    relevant_chunks:  list[SearchResult]     # graded-relevant chunks only
    web_results:      list[WebResult]        # Tavily results (if fallback)
    web_refined:      str                    # refined web content
    final_context:    str                    # context string fed to Gemini
    answer:           str                    # final generated answer
    source:           str                    # "qdrant" | "web"
    latency_ms:       float


# ── Response ───────────────────────────────────────────────────────────────────

@dataclass
class CRAGResponse:
    question: str
    answer: str
    source: str                              # "qdrant" | "web"
    relevant_chunks: list[SearchResult] = field(default_factory=list)
    web_results: list[WebResult] = field(default_factory=list)
    latency_ms: float = 0.0

    def print_summary(self):
        src_label = "Qdrant (graded relevant)" if self.source == "qdrant" else "Web search (fallback)"
        print(f"\n{'='*60}")
        print(f"QUESTION: {self.question}")
        print(f"SOURCE:   {src_label}")
        print(f"LATENCY:  {self.latency_ms:.0f}ms")
        print(f"{'='*60}")
        print(f"\nANSWER:\n{self.answer}")
        if self.relevant_chunks:
            print(f"\nGRADED RELEVANT CHUNKS ({len(self.relevant_chunks)}):")
            for c in self.relevant_chunks:
                print(f"  score={c.score:.3f} | {c.metadata.get('section','N/A')} | pg {c.metadata.get('page','?')}")
        if self.web_results:
            print(f"\nWEB FALLBACK SOURCES ({len(self.web_results)}):")
            for w in self.web_results:
                print(f"  {w.title} | {w.url}")
        print(f"{'='*60}\n")


# ── Node functions ─────────────────────────────────────────────────────────────

def build_crag_graph(
    embedder: GeminiEmbedder,
    store: QdrantStore,
    grader: RetrievalGrader,
    web_search: TavilyWebSearch,
    llm: GeminiLLM,
    top_k: int = 5,
) -> StateGraph:
    """
    Build the CRAG LangGraph. Returns a compiled graph ready to invoke.

    The graph has 5 nodes and 1 conditional edge:
      retrieve → grade → [conditional] → generate
                                       → web_search → generate
    """

    # ── Node 1: retrieve ──────────────────────────────────────────────────────
    def retrieve(state: CRAGState) -> dict:
        """Embed the query and retrieve top-k chunks from Qdrant."""
        logger.info(f"[CRAG] Node: retrieve | question='{state['question'][:50]}...'")

        query_vector = embedder.embed_query(state["question"])
        chunks = store.search(query_vector=query_vector, top_k=top_k)

        logger.info(f"[CRAG] Retrieved {len(chunks)} chunks | "
                    f"scores: {[round(c.score, 3) for c in chunks]}")

        return {
            "query_vector": query_vector,
            "retrieved_chunks": chunks,
        }

    # ── Node 2: grade ─────────────────────────────────────────────────────────
    def grade(state: CRAGState) -> dict:
        """
        Grade each retrieved chunk for relevance.
        Sets grade_decision to WEB_FALLBACK or USE_FILTERED.
        """
        logger.info(f"[CRAG] Node: grade | grading {len(state['retrieved_chunks'])} chunks")

        decision, relevant = grader.grade_all(
            question=state["question"],
            chunks=state["retrieved_chunks"],
        )

        logger.info(f"[CRAG] Grade decision: {decision} | "
                    f"{len(relevant)}/{len(state['retrieved_chunks'])} relevant")

        return {
            "grade_decision": decision,
            "relevant_chunks": relevant,
        }

    # ── Node 3: web_search (fallback) ─────────────────────────────────────────
    def web_search_node(state: CRAGState) -> dict:
        """
        Called only when grade_decision == WEB_FALLBACK.
        Searches Tavily and refines results into clean legal facts.
        """
        logger.info(f"[CRAG] Node: web_search | all chunks irrelevant, searching web")

        results, refined = web_search.search_and_refine(state["question"])

        return {
            "web_results": results,
            "web_refined": refined,
            "final_context": f"[Web search results — refined]\n\n{refined}",
            "source": "web",
        }

    # ── Node 4: prepare_context (after grading, before generate) ──────────────
    def prepare_context(state: CRAGState) -> dict:
        """
        Called only when grade_decision == USE_FILTERED.
        Formats graded-relevant chunks into context string.
        """
        logger.info(f"[CRAG] Node: prepare_context | "
                    f"using {len(state['relevant_chunks'])} graded chunks")

        context = format_context(state["relevant_chunks"])
        return {
            "final_context": context,
            "source": "qdrant",
        }

    # ── Node 5: generate ──────────────────────────────────────────────────────
    def generate(state: CRAGState) -> dict:
        """Generate the final answer using verified context."""
        logger.info(f"[CRAG] Node: generate | source={state.get('source', '?')}")

        prompt = SYNTHESIS_PROMPT.format(
            context=state["final_context"],
            question=state["question"],
        )
        answer = llm.generate(prompt, temperature=0.2)

        logger.info(f"[CRAG] Answer generated ({len(answer)} chars)")
        return {"answer": answer}

    # ── Conditional router ────────────────────────────────────────────────────
    def route_after_grade(state: CRAGState) -> Literal["web_search_node", "prepare_context"]:
        """
        KEY CONCEPT: Conditional edges in LangGraph
        ────────────────────────────────────────────
        A conditional edge is a function that looks at the current state
        and returns the NAME of the next node to go to.

        This is how CRAG implements the branching logic:
          if grade_decision == WEB_FALLBACK → go to web_search_node
          if grade_decision == USE_FILTERED → go to prepare_context

        In LangGraph, this function is registered with add_conditional_edges().
        The return value must exactly match a node name in the graph.
        """
        if state["grade_decision"] == RetrievalGrader.WEB_FALLBACK:
            logger.info("[CRAG] Routing → web_search_node")
            return "web_search_node"
        else:
            logger.info("[CRAG] Routing → prepare_context")
            return "prepare_context"

    # ── Build graph ────────────────────────────────────────────────────────────
    graph = StateGraph(CRAGState)

    # Add nodes
    graph.add_node("retrieve",        retrieve)
    graph.add_node("grade",           grade)
    graph.add_node("web_search_node", web_search_node)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("generate",        generate)

    # Add edges (fixed flow)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")

    # Add conditional edge — the CRAG branching logic
    graph.add_conditional_edges(
        "grade",                      # from this node
        route_after_grade,            # call this function to decide
        {
            "web_search_node": "web_search_node",   # if function returns "web_search_node"
            "prepare_context": "prepare_context",   # if function returns "prepare_context"
        }
    )

    # Both paths converge at generate
    graph.add_edge("web_search_node", "generate")
    graph.add_edge("prepare_context", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ── CRAG chain class ───────────────────────────────────────────────────────────

class CRAGChain:
    """
    High-level interface for the CRAG pipeline.
    Initializes all components and exposes a single .query() method.
    """

    def __init__(self, top_k: int = 5):
        logger.info("Initializing CRAG chain...")
        self.embedder   = GeminiEmbedder()
        self.store      = QdrantStore()
        self.grader     = RetrievalGrader()
        self.llm        = GeminiLLM()

        # Web search is optional — CRAG still works without it
        # (falls back to "insufficient context" message)
        try:
            self.web_search = TavilyWebSearch()
        except ValueError:
            logger.warning("Tavily API key not set — web fallback disabled")
            self.web_search = None

        self.graph = build_crag_graph(
            embedder=self.embedder,
            store=self.store,
            grader=self.grader,
            web_search=self.web_search,
            llm=self.llm,
            top_k=top_k,
        )
        logger.success("CRAG chain ready")

    def query(self, question: str) -> CRAGResponse:
        """Run the full CRAG pipeline for a question."""
        start = time.time()

        # Initial state — only question is set; all other fields start empty
        initial_state: CRAGState = {
            "question":         question,
            "query_vector":     [],
            "retrieved_chunks": [],
            "grade_decision":   "",
            "relevant_chunks":  [],
            "web_results":      [],
            "web_refined":      "",
            "final_context":    "",
            "answer":           "",
            "source":           "",
            "latency_ms":       0.0,
        }

        final_state = self.graph.invoke(initial_state)
        latency_ms = (time.time() - start) * 1000

        return CRAGResponse(
            question=question,
            answer=final_state["answer"],
            source=final_state.get("source", "unknown"),
            relevant_chunks=final_state.get("relevant_chunks", []),
            web_results=final_state.get("web_results", []),
            latency_ms=latency_ms,
        )


# ── Compare CRAG vs Naive ──────────────────────────────────────────────────────

def compare_with_naive(question: str) -> None:
    """
    Run the same question through both Naive RAG and CRAG side by side.
    Useful for seeing the improvement CRAG provides on failure cases.
    """
    from patterns.naive_rag import NaiveRAG

    print(f"\n{'='*60}")
    print(f"COMPARISON: Naive RAG vs CRAG")
    print(f"Question: {question}")
    print(f"{'='*60}")

    print("\n--- Naive RAG ---")
    naive = NaiveRAG(top_k=5)
    naive_resp = naive.query(question)
    print(f"Top retrieval score: {naive_resp.top_score:.3f}")
    print(f"Answer preview: {naive_resp.answer[:300]}...")

    print("\n--- CRAG ---")
    crag = CRAGChain(top_k=5)
    crag_resp = crag.query(question)
    print(f"Source: {crag_resp.source}")
    print(f"Relevant chunks kept: {len(crag_resp.relevant_chunks)}")
    print(f"Answer preview: {crag_resp.answer[:300]}...")


# ── Interactive demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: python -m patterns.crag

    Good test cases (from Phase 2 failure analysis):
      - "Under what circumstances is a person NOT liable for murder?" (Type D)
      - "What did the Supreme Court rule in the Kesavananda Bharati case?" (Type E)

    Watch the logs:
      - Grade: YES/NO for each chunk
      - Routing decision: prepare_context vs web_search_node
      - Source in the final response: qdrant vs web
    """
    import sys

    crag = CRAGChain(top_k=5)

    # If a question is passed as argument, run it directly
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        resp = crag.query(question)
        resp.print_summary()
        sys.exit(0)

    print("\nLexRAG — CRAG Demo (Phase 3)")
    print("Type 'compare <question>' to compare with Naive RAG")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("Your legal question: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        if user_input.lower().startswith("compare "):
            question = user_input[8:].strip()
            compare_with_naive(question)
        else:
            resp = crag.query(user_input)
            resp.print_summary()
