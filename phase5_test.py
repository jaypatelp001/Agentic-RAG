"""
patterns/phase5_test.py
────────────────────────────────────────────────────────────────────────────────
All 4 patterns head-to-head on every Phase 2 failure case.

Run: python -m patterns.phase5_test

This is the capstone test of Phases 2-5.
Shows exactly which pattern fixes which failure type.

EXPECTED RESULTS MATRIX:
  Question Type  | Naive | CRAG | Self-RAG | RAG-Fusion | Adaptive
  ───────────────┼───────┼──────┼──────────┼────────────┼─────────
  A: Simple      |  OK   |  OK  |   OK     |    OK      |   OK
  B: Multi-hop   |  FAIL |  ~   |   ~      |    OK      |   ~
  C: Comparative |  FAIL |  ~   |   ~      |    OK      |   ~
  D: Negation    |  FAIL |  OK  |   OK     |    ~       |   ~
  E: Out-corpus  |  FAIL |  OK  |   OK     |    FAIL    |   FAIL
  F: Structured  |  FAIL |  ~   |   ~      |    ~       |   OK

~ = partially better than naive, but not the best pattern for that type
"""

import time
from naive_rag     import NaiveRAG
from crag          import CRAGChain
from self_rag      import SelfRAGChain
from rag_fusion    import RAGFusionChain
from adaptive_rag  import AdaptiveRAGChain

TEST_CASES = [
    {
        "id": "A1", "type": "simple",
        "question": "What is the punishment for murder under the Indian Penal Code?",
        "best_pattern": "All",
    },
    {
        "id": "B1", "type": "multi_hop",
        "question": "What is the difference between murder under Section 302 and culpable homicide under Section 304 of the IPC?",
        "best_pattern": "RAG-Fusion",
    },
    {
        "id": "C1", "type": "comparative",
        "question": "How do the exemptions to RTI disclosure compare to the freedom of speech under Article 19 of the Constitution?",
        "best_pattern": "RAG-Fusion",
    },
    {
        "id": "D1", "type": "negation",
        "question": "Under what circumstances is a person NOT liable for murder even if they cause death?",
        "best_pattern": "CRAG + Self-RAG",
    },
    {
        "id": "E1", "type": "out_of_corpus",
        "question": "What did the Supreme Court rule in the Kesavananda Bharati case?",
        "best_pattern": "CRAG (web fallback)",
    },
    {
        "id": "F1", "type": "structured",
        "question": "List all sections in the Indian Penal Code that mention the word fine as punishment",
        "best_pattern": "Adaptive RAG (SQL)",
    },
]


def run_all_patterns(question: str, patterns: dict) -> dict:
    """Run one question through all patterns and collect results."""
    results = {}
    for name, chain in patterns.items():
        t0 = time.time()
        try:
            resp = chain.query(question)
            latency = (time.time() - t0) * 1000
            # Extract the answer text regardless of response type
            answer = getattr(resp, "answer", str(resp))
            results[name] = {
                "answer_preview": answer[:200] + "...",
                "latency_ms": round(latency),
                "ok": True,
            }
            # Add pattern-specific metadata
            if hasattr(resp, "source"):
                results[name]["source"] = resp.source
            if hasattr(resp, "route"):
                results[name]["route"] = resp.route
            if hasattr(resp, "regenerations"):
                results[name]["regenerations"] = resp.regenerations
            if hasattr(resp, "query_variants"):
                results[name]["n_variants"] = len(resp.query_variants)
        except Exception as e:
            results[name] = {"ok": False, "error": str(e), "latency_ms": 0}
    return results


def print_comparison(case: dict, results: dict):
    print(f"\n{'='*68}")
    print(f"[{case['id']}] TYPE: {case['type'].upper()}")
    print(f"Q: {case['question']}")
    print(f"Best pattern: {case['best_pattern']}")
    print(f"{'─'*68}")
    for name, r in results.items():
        if not r["ok"]:
            print(f"  {name:12s} ERROR: {r['error'][:60]}")
            continue
        meta = []
        if "source"        in r: meta.append(f"src={r['source']}")
        if "route"         in r: meta.append(f"route={r['route']}")
        if "regenerations" in r: meta.append(f"regen={r['regenerations']}")
        if "n_variants"    in r: meta.append(f"variants={r['n_variants']}")
        meta_str = " | ".join(meta)
        print(f"  {name:12s} {r['latency_ms']:5d}ms  {meta_str}")
        print(f"               {r['answer_preview'][:120]}...")
    print(f"{'='*68}")


def main():
    print("\nInitializing all 4 pattern chains...")
    patterns = {
        "Naive":      NaiveRAG(top_k=5),
        "CRAG":       CRAGChain(top_k=5),
        "Self-RAG":   SelfRAGChain(top_k=5),
        "RAG-Fusion": RAGFusionChain(n_variants=3, top_k=5),
        "Adaptive":   AdaptiveRAGChain(top_k=5),
    }
    print("All chains ready.\n")

    for case in TEST_CASES:
        print(f"\nRunning [{case['id']}]: {case['question'][:60]}...")
        results = run_all_patterns(case["question"], patterns)
        print_comparison(case, results)
        time.sleep(1)  # rate limit between questions

    print("\nPhase 5 complete. All 4 patterns tested across all failure types.")
    print("Ready for Phase 6: RAGAS evaluation + LangFuse + Redis + Docker.")


if __name__ == "__main__":
    main()
