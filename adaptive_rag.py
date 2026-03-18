"""
patterns/adaptive_rag.py
────────────────────────────────────────────────────────────────────────────────
Adaptive RAG — routes each query to the right retrieval strategy at runtime.

KEY CONCEPT: Vector search vs SQL — when to use each
──────────────────────────────────────────────────────
Vector search (Qdrant) is best for:
  "What does Section 302 say about punishment for murder?"  → semantic
  "Explain the right to information under RTI Act"          → conceptual
  "What are the general exceptions to IPC liability?"       → exploratory

SQL (SQLite) is best for:
  "List all sections of the IPC that mention the word 'fine'"    → filter
  "How many sections are in the RTI Act?"                        → count
  "Give me all sections from Chapter IV of the IPC"             → structured
  "Which section covers Section 420?"                            → exact lookup

Hybrid (both) is best for:
  "List all bail-related sections of CrPC and explain each one" → list + explain

THE ROUTER:
  A Gemini call with temperature=0.0 classifies the question as:
    "vector"  → go to Qdrant
    "sql"     → go to SQLite
    "hybrid"  → run both, merge results, generate from combined context

SQL QUERY GENERATION:
  The router also generates the SQL query string.
  We use parameterized-style generation (Gemini writes the WHERE clause).
  The schema is injected into the prompt so Gemini knows the column names.
  Results are truncated to top_k rows to avoid context overflow.
"""

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Literal, TypedDict
from loguru import logger
from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, END

from llm import GeminiLLM
from prompts import ADAPTIVE_ROUTER_PROMPT, SYNTHESIS_PROMPT, format_context
from embedder import GeminiEmbedder
from qdrant_store import QdrantStore, SearchResult

load_dotenv()


# ── SQL query generator ────────────────────────────────────────────────────────

SQL_SCHEMA = """
Table: chunks
Columns:
  chunk_id    TEXT  -- unique ID e.g. "ipc_1860.pdf_p47_c2"
  source      TEXT  -- filename e.g. "ipc_1860.pdf"
  act_name    TEXT  -- e.g. "Indian Penal Code", "Right to Information Act 2005"
  page        INT   -- page number in source PDF
  section     TEXT  -- e.g. "Section 302", "Article 19" (may be NULL)
  chunk_index INT   -- chunk number within page
  chunk_size  INT   -- character count
  text        TEXT  -- full chunk text
"""

SQL_GEN_PROMPT = """\
You are a SQL query generator for a legal document database.

DATABASE SCHEMA:
{schema}

USER QUESTION: {question}

Write a valid SQLite SELECT query that retrieves relevant rows to answer this question.
Rules:
  - Always SELECT: chunk_id, source, act_name, page, section, text
  - Use LIKE for text search: text LIKE '%keyword%'
  - Use = for exact match: act_name = 'Indian Penal Code'
  - Add LIMIT {top_k} at the end
  - Return ONLY the SQL query, nothing else

SQL QUERY:
"""


class SQLRetriever:
    """Generates and executes SQL queries against the SQLite metadata DB."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.getenv("SQLITE_DB_PATH", "./data/lexrag_metadata.db")
        self.llm = GeminiLLM()

    def generate_sql(self, question: str, top_k: int = 10) -> str:
        """Ask Gemini to write the SQL query for this question."""
        prompt = SQL_GEN_PROMPT.format(
            schema=SQL_SCHEMA,
            question=question,
            top_k=top_k,
        )
        raw = self.llm.generate(prompt, temperature=0.0, max_tokens=256)

        # Strip markdown code fences if present
        sql = raw.strip()
        if sql.startswith("```"):
            sql = "\n".join(sql.split("\n")[1:-1])
        sql = sql.strip().rstrip(";")

        logger.debug(f"  Generated SQL: {sql[:120]}...")
        return sql

    def execute(self, sql: str) -> list[dict]:
        """Execute SQL and return rows as dicts."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            logger.info(f"  SQL returned {len(rows)} rows")
            return rows
        except Exception as e:
            logger.error(f"  SQL execution failed: {e}\n  SQL was: {sql}")
            return []

    def rows_to_search_results(self, rows: list[dict]) -> list[SearchResult]:
        """Convert SQL rows to SearchResult objects for unified handling."""
        results = []
        for row in rows:
            results.append(SearchResult(
                text=row.get("text", ""),
                score=1.0,   # SQL results have no cosine score — use 1.0 as placeholder
                metadata={
                    "chunk_id":  row.get("chunk_id"),
                    "source":    row.get("source"),
                    "act_name":  row.get("act_name"),
                    "page":      row.get("page"),
                    "section":   row.get("section"),
                    "retrieval": "sql",
                }
            ))
        return results

    def retrieve(self, question: str, top_k: int = 10) -> list[SearchResult]:
        """Full SQL retrieval: generate query → execute → convert to SearchResult."""
        logger.info(f"[Adaptive] SQL retrieval for: '{question[:60]}...'")
        sql = self.generate_sql(question, top_k)
        rows = self.execute(sql)
        results = self.rows_to_search_results(rows)
        return results


# ── State ──────────────────────────────────────────────────────────────────────

class AdaptiveState(TypedDict):
    question:       str
    route:          str          # "vector" | "sql" | "hybrid"
    vector_results: list[SearchResult]
    sql_results:    list[SearchResult]
    context_str:    str
    answer:         str
    latency_ms:     float


# ── Response ───────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveResponse:
    question:       str
    answer:         str
    route:          str
    vector_results: list[SearchResult] = field(default_factory=list)
    sql_results:    list[SearchResult] = field(default_factory=list)
    latency_ms:     float = 0.0

    def print_summary(self):
        print(f"\n{'='*62}")
        print(f"QUESTION: {self.question}")
        print(f"ROUTE:    {self.route.upper()}")
        print(f"LATENCY:  {self.latency_ms:.0f}ms")
        if self.vector_results:
            print(f"\nVECTOR RESULTS ({len(self.vector_results)}):")
            for r in self.vector_results[:3]:
                print(f"  score={r.score:.3f} | {r.metadata.get('section','N/A')}")
        if self.sql_results:
            print(f"\nSQL RESULTS ({len(self.sql_results)} rows):")
            for r in self.sql_results[:3]:
                print(f"  {r.metadata.get('act_name')} | pg {r.metadata.get('page')} | {r.metadata.get('section','N/A')}")
        print(f"\nANSWER:\n{self.answer}")
        print(f"{'='*62}\n")


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_adaptive_graph(
    embedder: GeminiEmbedder,
    store: QdrantStore,
    sql_retriever: SQLRetriever,
    llm: GeminiLLM,
    top_k: int = 5,
) -> StateGraph:

    # ── Node 1: route ──────────────────────────────────────────────────────────
    def route(state: AdaptiveState) -> dict:
        """
        Classify the question as vector / sql / hybrid.
        temperature=0.0 for deterministic routing.
        """
        logger.info(f"[Adaptive] Node: route")
        prompt = ADAPTIVE_ROUTER_PROMPT.format(question=state["question"])
        raw = llm.grade(prompt)   # temp=0.0, short output

        # Normalize
        if "hybrid" in raw:
            decision = "hybrid"
        elif "sql" in raw:
            decision = "sql"
        else:
            decision = "vector"  # default fallback

        logger.info(f"[Adaptive] Route decision: {decision.upper()}")
        return {"route": decision}

    # ── Node 2: vector_search ──────────────────────────────────────────────────
    def vector_search(state: AdaptiveState) -> dict:
        """Standard Qdrant semantic search."""
        logger.info("[Adaptive] Node: vector_search")
        qv = embedder.embed_query(state["question"])
        results = store.search(query_vector=qv, top_k=top_k)
        logger.info(f"  Vector: {len(results)} results")
        return {"vector_results": results}

    # ── Node 3: sql_search ─────────────────────────────────────────────────────
    def sql_search(state: AdaptiveState) -> dict:
        """LLM-generated SQL query against SQLite metadata DB."""
        logger.info("[Adaptive] Node: sql_search")
        results = sql_retriever.retrieve(state["question"], top_k=top_k * 2)
        return {"sql_results": results}

    # ── Node 4: merge_context ──────────────────────────────────────────────────
    def merge_context(state: AdaptiveState) -> dict:
        """
        Merge vector and SQL results based on routing decision.

        vector → use only vector results
        sql    → use only SQL results
        hybrid → interleave: vector results first, SQL results appended
                 The generator sees both semantic context and structured data.
        """
        logger.info("[Adaptive] Node: merge_context")
        route = state["route"]

        if route == "vector":
            combined = state["vector_results"]
        elif route == "sql":
            combined = state["sql_results"]
        else:  # hybrid
            # Interleave: alternate vector and SQL results
            v, s = state["vector_results"], state["sql_results"]
            combined = []
            for i in range(max(len(v), len(s))):
                if i < len(v): combined.append(v[i])
                if i < len(s): combined.append(s[i])
            combined = combined[:top_k * 2]

        context_str = format_context(combined)
        logger.info(f"  Context built from {len(combined)} results ({route} route)")
        return {"context_str": context_str}

    # ── Node 5: generate ───────────────────────────────────────────────────────
    def generate(state: AdaptiveState) -> dict:
        logger.info("[Adaptive] Node: generate")
        prompt = SYNTHESIS_PROMPT.format(
            context=state["context_str"],
            question=state["question"],
        )
        answer = llm.generate(prompt, temperature=0.2)
        return {"answer": answer}

    # ── Conditional routing ────────────────────────────────────────────────────
    def route_after_decision(
        state: AdaptiveState,
    ) -> Literal["vector_search", "sql_search", "both_searches"]:
        """
        After route node:
          vector → go to vector_search only
          sql    → go to sql_search only
          hybrid → go to both (we model this by going to vector_search first,
                   then sql_search, then merge)

        NOTE: True parallel hybrid requires LangGraph's Send() API.
        Here we sequence vector → sql → merge for simplicity.
        """
        r = state["route"]
        if r == "sql":
            return "sql_search"
        return "vector_search"  # handles both "vector" and "hybrid"

    def route_after_vector(
        state: AdaptiveState,
    ) -> Literal["sql_search", "merge_context"]:
        """After vector search: hybrid needs SQL too, vector goes straight to merge."""
        if state["route"] == "hybrid":
            return "sql_search"
        return "merge_context"

    # ── Build graph ────────────────────────────────────────────────────────────
    graph = StateGraph(AdaptiveState)

    graph.add_node("route",         route)
    graph.add_node("vector_search", vector_search)
    graph.add_node("sql_search",    sql_search)
    graph.add_node("merge_context", merge_context)
    graph.add_node("generate",      generate)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        route_after_decision,
        {"vector_search": "vector_search", "sql_search": "sql_search"},
    )
    graph.add_conditional_edges(
        "vector_search",
        route_after_vector,
        {"sql_search": "sql_search", "merge_context": "merge_context"},
    )
    graph.add_edge("sql_search",    "merge_context")
    graph.add_edge("merge_context", "generate")
    graph.add_edge("generate",      END)

    return graph.compile()


# ── Adaptive chain class ───────────────────────────────────────────────────────

class AdaptiveRAGChain:
    """High-level interface for Adaptive RAG."""

    def __init__(self, top_k: int = 5):
        logger.info("Initializing Adaptive RAG chain...")
        self.embedder      = GeminiEmbedder()
        self.store         = QdrantStore()
        self.sql_retriever = SQLRetriever()
        self.llm           = GeminiLLM()
        self.graph         = build_adaptive_graph(
            embedder=self.embedder,
            store=self.store,
            sql_retriever=self.sql_retriever,
            llm=self.llm,
            top_k=top_k,
        )
        logger.success("Adaptive RAG chain ready")

    def query(self, question: str) -> AdaptiveResponse:
        start = time.time()
        initial: AdaptiveState = {
            "question":       question,
            "route":          "",
            "vector_results": [],
            "sql_results":    [],
            "context_str":    "",
            "answer":         "",
            "latency_ms":     0.0,
        }
        final = self.graph.invoke(initial)
        latency_ms = (time.time() - start) * 1000
        return AdaptiveResponse(
            question=question,
            answer=final["answer"],
            route=final["route"],
            vector_results=final.get("vector_results", []),
            sql_results=final.get("sql_results", []),
            latency_ms=latency_ms,
        )


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: python -m patterns.adaptive_rag

    Test each route type:
      SQL:    "List all sections in the IPC that mention the word fine"
      Vector: "What is the legal definition of abetment under IPC?"
      Hybrid: "List all bail sections in CrPC and explain what each covers"
    """
    chain = AdaptiveRAGChain(top_k=5)

    test_cases = [
        ("SQL route",    "List all sections in the Indian Penal Code that mention the word fine as punishment"),
        ("Vector route", "What is the legal definition of abetment under the Indian Penal Code?"),
        ("Hybrid route", "List all sections related to bail in CrPC and explain the conditions for each"),
    ]

    for label, question in test_cases:
        print(f"\n{'='*62}")
        print(f"TEST: {label}")
        resp = chain.query(question)
        resp.print_summary()
