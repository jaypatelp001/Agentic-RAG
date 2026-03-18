"""
patterns/crag_test.py
────────────────────────────────────────────────────────────────────────────────
Test CRAG against the specific Phase 2 failure cases.

Run: python -m patterns.crag_test

Tests the 4 questions that Naive RAG failed on:
  - Type D1: Negation — "NOT liable for murder"
  - Type D2: Negation — RTI exemptions
  - Type E1: Out-of-corpus — SEBI + IPC
  - Type E2: Out-of-corpus — Kesavananda Bharati case

For each question, prints side-by-side:
  Naive RAG: top score, diagnosis, answer preview
  CRAG:      source (qdrant/web), relevant chunks kept, answer preview
"""

from patterns.naive_rag import NaiveRAG
from patterns.crag import CRAGChain

FAILURE_CASES = [
    {
        "id": "D1",
        "question": "Under what circumstances is a person NOT liable for murder even if they cause death?",
        "why_naive_fails": "Embedding of 'NOT liable' retrieves murder punishment chunks (Section 302), not exceptions (Sections 76-106)",
    },
    {
        "id": "D2",
        "question": "Which categories of information are exempt from disclosure under the RTI Act?",
        "why_naive_fails": "Retrieves general RTI chunks instead of Section 8 exemptions specifically",
    },
    {
        "id": "E1",
        "question": "What are the penalties for insider trading under SEBI regulations and how do they interact with IPC provisions on cheating?",
        "why_naive_fails": "SEBI regulations not in corpus — Gemini hallucinates from training data",
    },
    {
        "id": "E2",
        "question": "What did the Supreme Court rule in the Kesavananda Bharati case regarding the basic structure doctrine?",
        "why_naive_fails": "Case law not in corpus — Gemini very confidently hallucinates from training data",
    },
]


def run_comparison():
    print("\n" + "="*65)
    print("CRAG vs NAIVE RAG — Phase 2 Failure Cases")
    print("="*65)

    naive = NaiveRAG(top_k=5)
    crag  = CRAGChain(top_k=5)

    for case in FAILURE_CASES:
        print(f"\n[{case['id']}] {case['question'][:70]}...")
        print(f"Why naive fails: {case['why_naive_fails']}")
        print("-"*65)

        # Naive RAG
        naive_resp = naive.query(case["question"])
        print(f"NAIVE RAG:")
        print(f"  Top score:    {naive_resp.top_score:.3f}")
        print(f"  Avg score:    {naive_resp.avg_score:.3f}")
        print(f"  Answer:       {naive_resp.answer[:200]}...")

        print()

        # CRAG
        crag_resp = crag.query(case["question"])
        print(f"CRAG:")
        print(f"  Source:       {crag_resp.source}")
        print(f"  Relevant chunks kept: {len(crag_resp.relevant_chunks)}")
        if crag_resp.web_results:
            print(f"  Web sources:  {[w.title[:40] for w in crag_resp.web_results]}")
        print(f"  Answer:       {crag_resp.answer[:200]}...")
        print("="*65)


if __name__ == "__main__":
    run_comparison()
