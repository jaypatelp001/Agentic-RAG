"""
patterns/self_rag_test.py
────────────────────────────────────────────────────────────────────────────────
Compare CRAG vs Self-RAG on questions that stress-test different failure modes.

Run: python -m patterns.self_rag_test

What to look for:
  - Self-RAG regeneration_count > 0  → it caught and fixed a bad answer
  - Self-RAG used_context=False      → IsREL graded all chunks irrelevant,
                                       fell back to parametric memory
  - CRAG source='web'                → Tavily fallback triggered
  - Latency difference               → Self-RAG adds 2-4 extra LLM calls vs CRAG
"""

import time
from patterns.crag import CRAGChain
from patterns.self_rag import SelfRAGChain

TEST_CASES = [
    {
        "id": "1",
        "label": "Grounding test — Self-RAG should trigger IsSUP",
        "question": "Explain the constitutional validity of the RTI Act's exemptions under Section 8.",
        "expect": "Self-RAG may regenerate (Gemini adds constitutional details beyond RTI corpus)",
    },
    {
        "id": "2",
        "label": "Simple lookup — both should work, compare latency",
        "question": "What is the punishment for theft under the Indian Penal Code?",
        "expect": "Both correct. Self-RAG adds ~1-2s overhead for reflection tokens.",
    },
    {
        "id": "3",
        "label": "IsUSE test — answer that's true but doesn't address the question",
        "question": "What specific procedure must a Public Information Officer follow when refusing an RTI request?",
        "expect": "Self-RAG IsUSE may fail if Gemini gives general RTI background instead of procedure",
    },
    {
        "id": "4",
        "label": "IsRETRIEVE=No test — question LLM knows directly",
        "question": "In what year was the Indian Penal Code enacted?",
        "expect": "Self-RAG IsRETRIEVE should return NO → skip retrieval, answer directly (1860)",
    },
]


def run_comparison():
    print("\n" + "="*65)
    print("CRAG vs Self-RAG — Phase 4 Comparison")
    print("="*65)

    crag      = CRAGChain(top_k=5)
    self_rag  = SelfRAGChain(top_k=5)

    for case in TEST_CASES:
        print(f"\n[{case['id']}] {case['label']}")
        print(f"Q: {case['question']}")
        print(f"Expect: {case['expect']}")
        print("-"*65)

        # CRAG
        t0 = time.time()
        cr = crag.query(case["question"])
        crag_ms = (time.time() - t0) * 1000

        print(f"CRAG ({crag_ms:.0f}ms):")
        print(f"  source={cr.source} | relevant_chunks={len(cr.relevant_chunks)}")
        print(f"  answer: {cr.answer[:180]}...")

        print()

        # Self-RAG
        t0 = time.time()
        sr = self_rag.query(case["question"])
        srag_ms = (time.time() - t0) * 1000

        print(f"Self-RAG ({srag_ms:.0f}ms):")
        print(f"  used_context={sr.used_context} | regenerations={sr.regenerations}")
        if sr.fail_reasons:
            print(f"  fail_reasons={sr.fail_reasons}")
        print(f"  answer: {sr.answer[:180]}...")

        overhead = srag_ms - crag_ms
        print(f"\n  Overhead vs CRAG: +{overhead:.0f}ms for {sr.regenerations} regen(s) "
              f"+ {4} reflection token calls")
        print("="*65)


if __name__ == "__main__":
    run_comparison()
