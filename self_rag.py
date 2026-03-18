"""
patterns/self_rag.py
────────────────────────────────────────────────────────────────────────────────
Self-RAG — full LangGraph implementation with reflection loop.

KEY CONCEPT: Loops in LangGraph
─────────────────────────────────
CRAG was a DAG — each node ran exactly once, flow was one-directional.
Self-RAG introduces a loop: the generate node can route back to itself.

LangGraph supports cycles natively. The conditional edge after generate
can point BACK to generate (with an updated prompt), creating:

  retrieve → is_retrieve → [retrieve | direct_answer]
                               ↓
                          is_relevant → [use_context | use_memory]
                               ↓
                            generate
                               ↓
                         post_checks → [END | generate]  ← loop back
                                              ↑___________|

LOOP SAFETY:
  The state carries a `regeneration_count` field.
  The conditional edge after generate checks:
    if regeneration_count >= MAX_REGENERATE → force END (return best answer so far)
  This prevents infinite loops even if IsSUP/IsUSE keep failing.

GRAPH STATE DESIGN:
  New fields vs CRAG state:
    - use_context: bool     → True = use Qdrant chunks, False = use parametric memory
    - draft_answer: str     → current generation (may be replaced)
    - regeneration_count    → loop iteration counter
    - fail_reason: str      → which token failed and why (for debugging)
    - regeneration_prompt   → modified prompt used on retry (stricter instructions)
"""

import time
from dataclasses import dataclass, field
from typing import Literal, TypedDict
from loguru import logger

from langgraph.graph import StateGraph, END

from llm import GeminiLLM
from prompts import NAIVE_RAG_PROMPT, SYNTHESIS_PROMPT, format_context
from embedder import GeminiEmbedder
from self_rag_tokens import ReflectionTokens
from qdrant_store import QdrantStore, SearchResult


# ── State ──────────────────────────────────────────────────────────────────────

class SelfRAGState(TypedDict):
    question:            str
    query_vector:        list[float]
    retrieved_chunks:    list[SearchResult]
    relevant_chunks:     list[SearchResult]
    use_context:         bool        # True = use Qdrant chunks; False = LLM memory
    context_str:         str         # formatted context for generation
    draft_answer:        str         # current generated answer
    regeneration_count:  int         # how many times we've regenerated
    fail_reason:         str         # which token failed
    final_answer:        str         # confirmed good answer
    latency_ms:          float


MAX_REGENERATE = 2   # stop regenerating after this many attempts


# ── Response ───────────────────────────────────────────────────────────────────

@dataclass
class SelfRAGResponse:
    question: str
    answer: str
    used_context: bool
    relevant_chunks: list[SearchResult] = field(default_factory=list)
    regenerations: int = 0
    fail_reasons: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def print_summary(self):
        src = "Qdrant context" if self.used_context else "Parametric memory (no context)"
        print(f"\n{'='*62}")
        print(f"QUESTION:      {self.question}")
        print(f"SOURCE:        {src}")
        print(f"REGENERATIONS: {self.regenerations}")
        print(f"LATENCY:       {self.latency_ms:.0f}ms")
        if self.fail_reasons:
            print(f"FAIL REASONS:  {self.fail_reasons}")
        print(f"{'='*62}")
        print(f"\nANSWER:\n{self.answer}")
        if self.relevant_chunks:
            print(f"\nRELEVANT CHUNKS ({len(self.relevant_chunks)}):")
            for c in self.relevant_chunks:
                print(f"  score={c.score:.3f} | {c.metadata.get('section','N/A')}")
        print(f"{'='*62}\n")


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_self_rag_graph(
    embedder: GeminiEmbedder,
    store: QdrantStore,
    tokens: ReflectionTokens,
    llm: GeminiLLM,
    top_k: int = 5,
) -> StateGraph:
    """Build and compile the Self-RAG LangGraph."""

    # ── Node 1: check_retrieve ────────────────────────────────────────────────
    def check_retrieve(state: SelfRAGState) -> dict:
        """
        IsRETRIEVE token — decide whether retrieval is needed at all.
        If no, we'll generate directly from the LLM's parametric memory.
        """
        logger.info(f"[Self-RAG] Node: check_retrieve")
        should_retrieve = tokens.is_retrieve(state["question"])
        return {"use_context": should_retrieve}

    # ── Node 2: retrieve ──────────────────────────────────────────────────────
    def retrieve(state: SelfRAGState) -> dict:
        """Embed query and search Qdrant."""
        logger.info(f"[Self-RAG] Node: retrieve")
        qv = embedder.embed_query(state["question"])
        chunks = store.search(query_vector=qv, top_k=top_k)
        logger.info(f"[Self-RAG] Retrieved {len(chunks)} chunks")
        return {"query_vector": qv, "retrieved_chunks": chunks}

    # ── Node 3: check_relevance ───────────────────────────────────────────────
    def check_relevance(state: SelfRAGState) -> dict:
        """
        IsREL token — grade each chunk and keep only relevant ones.
        If none are relevant, set use_context=False (fall to parametric memory).
        """
        logger.info(f"[Self-RAG] Node: check_relevance")
        relevant = tokens.filter_relevant(state["question"], state["retrieved_chunks"])

        if not relevant:
            logger.warning("[Self-RAG] No relevant chunks — falling back to parametric memory")
            return {"relevant_chunks": [], "use_context": False, "context_str": ""}

        context_str = format_context(relevant)
        return {"relevant_chunks": relevant, "use_context": True, "context_str": context_str}

    # ── Node 4: generate ──────────────────────────────────────────────────────
    def generate(state: SelfRAGState) -> dict:
        """
        Generate an answer. On regeneration attempts, uses a stricter prompt
        that explicitly addresses the previous failure reason.
        """
        regen_count = state.get("regeneration_count", 0)
        fail_reason = state.get("fail_reason", "")
        logger.info(f"[Self-RAG] Node: generate (attempt {regen_count + 1})")

        if state["use_context"] and state.get("context_str"):
            # Use retrieved context
            if regen_count == 0:
                prompt = SYNTHESIS_PROMPT.format(
                    context=state["context_str"],
                    question=state["question"],
                )
            else:
                # Stricter regeneration prompt — explicitly mentions what failed
                strict_instruction = _build_regen_instruction(fail_reason)
                prompt = (
                    f"{strict_instruction}\n\n"
                    f"CONTEXT:\n{state['context_str']}\n\n"
                    f"QUESTION: {state['question']}\n\n"
                    f"ANSWER:"
                )
        else:
            # No context — generate from parametric memory
            prompt = (
                f"Answer this legal question about Indian law from your knowledge.\n"
                f"Be precise. Cite specific sections if you know them.\n"
                f"If uncertain, say so explicitly.\n\n"
                f"QUESTION: {state['question']}\n\nANSWER:"
            )

        answer = llm.generate(prompt, temperature=0.2 if regen_count == 0 else 0.1)
        logger.info(f"[Self-RAG] Draft answer generated ({len(answer)} chars)")
        return {"draft_answer": answer}

    # ── Node 5: post_checks ───────────────────────────────────────────────────
    def post_checks(state: SelfRAGState) -> dict:
        """
        Run IsSUP + IsUSE on the draft answer.
        If both pass → promote to final_answer.
        If either fails → record fail_reason, increment regeneration_count.
        The conditional edge after this node decides what happens next.
        """
        logger.info(f"[Self-RAG] Node: post_checks")
        context = state.get("context_str", "No context — parametric memory used.")
        answer = state["draft_answer"]

        passes, reason = tokens.passes_post_checks(
            question=state["question"],
            context=context,
            answer=answer,
        )

        regen_count = state.get("regeneration_count", 0)

        if passes:
            logger.success(f"[Self-RAG] Post-checks PASSED on attempt {regen_count + 1}")
            return {
                "final_answer": answer,
                "fail_reason": "",
                "regeneration_count": regen_count,
            }
        else:
            logger.warning(f"[Self-RAG] Post-checks FAILED: {reason} (attempt {regen_count + 1})")
            return {
                "final_answer": answer,   # keep as fallback even if failed
                "fail_reason": reason,
                "regeneration_count": regen_count + 1,
            }

    # ── Conditional edges ──────────────────────────────────────────────────────

    def route_after_retrieve_check(
        state: SelfRAGState,
    ) -> Literal["retrieve", "generate"]:
        """
        After IsRETRIEVE:
          use_context=True  → retrieve documents
          use_context=False → go directly to generate (no retrieval needed)
        """
        if state["use_context"]:
            return "retrieve"
        logger.info("[Self-RAG] Skipping retrieval — going to generate directly")
        return "generate"

    def route_after_post_checks(
        state: SelfRAGState,
    ) -> Literal["generate", "__end__"]:
        """
        After IsSUP + IsUSE:
          fail_reason == ""               → both passed → END
          regeneration_count >= MAX       → give up, return best answer → END
          otherwise                       → loop back to generate with stricter prompt

        KEY CONCEPT: This is the loop in LangGraph.
        Returning "generate" here sends the flow back to the generate node.
        The state carries regeneration_count so the generate node knows
        to use a stricter prompt on retry.
        """
        if not state.get("fail_reason"):
            logger.info("[Self-RAG] Post-checks passed → END")
            return END

        regen_count = state.get("regeneration_count", 0)
        if regen_count >= MAX_REGENERATE:
            logger.warning(
                f"[Self-RAG] Max regenerations ({MAX_REGENERATE}) reached → "
                f"returning best answer anyway"
            )
            return END

        logger.info(
            f"[Self-RAG] Looping back to generate "
            f"(attempt {regen_count + 1}/{MAX_REGENERATE})"
        )
        return "generate"

    # ── Build graph ────────────────────────────────────────────────────────────
    graph = StateGraph(SelfRAGState)

    graph.add_node("check_retrieve",    check_retrieve)
    graph.add_node("retrieve",          retrieve)
    graph.add_node("check_relevance",   check_relevance)
    graph.add_node("generate",          generate)
    graph.add_node("post_checks",       post_checks)

    graph.set_entry_point("check_retrieve")

    # Fixed edges
    graph.add_edge("retrieve",          "check_relevance")
    graph.add_edge("check_relevance",   "generate")
    graph.add_edge("generate",          "post_checks")

    # Conditional edge 1: after IsRETRIEVE
    graph.add_conditional_edges(
        "check_retrieve",
        route_after_retrieve_check,
        {"retrieve": "retrieve", "generate": "generate"},
    )

    # Conditional edge 2: after post_checks — this is the LOOP
    graph.add_conditional_edges(
        "post_checks",
        route_after_post_checks,
        {"generate": "generate", END: END},
    )

    return graph.compile()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_regen_instruction(fail_reason: str) -> str:
    """
    Build a targeted regeneration instruction based on which token failed.
    This is what makes the retry smarter than just re-running the same prompt.
    """
    if "IsSUP" in fail_reason:
        return (
            "CRITICAL: Your previous answer contained claims NOT found in the context. "
            "This time, ONLY state facts that are explicitly present in the context below. "
            "Do NOT add any information from your training data. "
            "If the context doesn't say it, don't include it."
        )
    elif "IsUSE" in fail_reason:
        return (
            "CRITICAL: Your previous answer did not directly address the question asked. "
            "Focus specifically on answering what was asked. "
            "Be concrete and specific. Do not provide background context unless directly relevant."
        )
    return "Please provide a more accurate and relevant answer based strictly on the context."


# ── Self-RAG chain class ───────────────────────────────────────────────────────

class SelfRAGChain:
    """High-level interface for the Self-RAG pipeline."""

    def __init__(self, top_k: int = 5):
        logger.info("Initializing Self-RAG chain...")
        self.embedder = GeminiEmbedder()
        self.store    = QdrantStore()
        self.tokens   = ReflectionTokens()
        self.llm      = GeminiLLM()
        self.graph    = build_self_rag_graph(
            embedder=self.embedder,
            store=self.store,
            tokens=self.tokens,
            llm=self.llm,
            top_k=top_k,
        )
        logger.success("Self-RAG chain ready")

    def query(self, question: str) -> SelfRAGResponse:
        start = time.time()

        initial: SelfRAGState = {
            "question":           question,
            "query_vector":       [],
            "retrieved_chunks":   [],
            "relevant_chunks":    [],
            "use_context":        True,
            "context_str":        "",
            "draft_answer":       "",
            "regeneration_count": 0,
            "fail_reason":        "",
            "final_answer":       "",
            "latency_ms":         0.0,
        }

        final = self.graph.invoke(initial)
        latency_ms = (time.time() - start) * 1000

        # Collect all fail reasons across regeneration attempts
        fail_reasons = []
        if final.get("fail_reason"):
            fail_reasons.append(final["fail_reason"])

        return SelfRAGResponse(
            question=question,
            answer=final["final_answer"] or final.get("draft_answer", ""),
            used_context=final.get("use_context", False),
            relevant_chunks=final.get("relevant_chunks", []),
            regenerations=final.get("regeneration_count", 0),
            fail_reasons=fail_reasons,
            latency_ms=latency_ms,
        )


# ── Interactive demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: python -m patterns.self_rag

    Watch for the reflection token decisions in the logs:
      [Self-RAG] IsRETRIEVE: RETRIEVE
      [Self-RAG] IsREL — grading 5 chunks...
      [Self-RAG] IsREL: 3/5 chunks relevant
      [Self-RAG] IsSUP: SUPPORTED
      [Self-RAG] IsUSE: USEFUL
      [Self-RAG] Post-checks PASSED on attempt 1

    Trigger regeneration by asking a question where the context is thin:
      "Explain the constitutional validity of the RTI Act's exemptions"
      → IsSUP may fail if Gemini adds constitutional law details
        not present in the RTI chunks
    """
    chain = SelfRAGChain(top_k=5)

    print("\nLexRAG — Self-RAG Demo (Phase 4)")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Your legal question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        resp = chain.query(question)
        resp.print_summary()
