"""
patterns/failure_analysis.py
────────────────────────────────────────────────────────────────────────────────
Deliberately stress-test Naive RAG with 10 hard legal questions.
Produces a failure report showing exactly where and why retrieval breaks.

Run: python -m patterns.failure_analysis

WHY THIS MATTERS:
─────────────────
Every advanced RAG pattern (CRAG, Self-RAG, RAG-Fusion) was invented to solve
a specific failure mode discovered by running exactly this kind of analysis.

The 10 questions below are chosen to trigger specific known failure modes:

  Type A — Simple lookup        → Naive RAG handles this fine (baseline OK)
  Type B — Multi-hop reasoning  → Retrieval gets one hop, misses the other
  Type C — Comparative question → Single query can't capture both sides
  Type D — Negation/exception   → Embedding similarity misses negation semantics
  Type E — Out-of-corpus        → Naive RAG hallucinates instead of admitting

After running this, you'll SEE the failure modes in the scores and answers.
That experience is the motivation for CRAG (Phase 3).
"""

import json
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from naive_rag import NaiveRAG, RAGResponse


# ── Test questions ─────────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    # ── Type A: Simple lookup (should work) ────────────────────────────────────
    {
        "id": "A1",
        "type": "simple_lookup",
        "question": "What is the punishment for murder under the Indian Penal Code?",
        "expected_section": "Section 302",
        "failure_mode": "None — baseline should handle this",
    },
    {
        "id": "A2",
        "type": "simple_lookup",
        "question": "What is the time limit for providing information under the RTI Act?",
        "expected_section": "Section 7",
        "failure_mode": "None — baseline should handle this",
    },

    # ── Type B: Multi-hop reasoning (will likely fail) ─────────────────────────
    {
        "id": "B1",
        "type": "multi_hop",
        "question": (
            "If a person commits culpable homicide while under the influence of alcohol "
            "provided by another person, what sections of the IPC apply to both parties?"
        ),
        "expected_section": "Section 299, 304, 328",
        "failure_mode": "Multi-hop: retrieval captures one section but misses the intoxication angle",
    },
    {
        "id": "B2",
        "type": "multi_hop",
        "question": (
            "What is the procedure for filing an appeal if a Public Information Officer "
            "under RTI fails to respond within the stipulated time?"
        ),
        "expected_section": "Section 19",
        "failure_mode": "Multi-hop: query embeds to RTI basics, misses the appeal procedure chunk",
    },

    # ── Type C: Comparative (will fail — single query can't cover both sides) ──
    {
        "id": "C1",
        "type": "comparative",
        "question": (
            "What is the difference between murder (Section 302) and culpable homicide "
            "not amounting to murder (Section 304) under IPC?"
        ),
        "expected_section": "Section 299, 300, 302, 304",
        "failure_mode": "Comparative: retrieves chunks for one side, not both — RAG-Fusion fixes this",
    },
    {
        "id": "C2",
        "type": "comparative",
        "question": (
            "How do the exemptions to disclosure under the RTI Act compare to "
            "the freedom of speech guaranteed under Article 19 of the Constitution?"
        ),
        "expected_section": "RTI Section 8, Constitution Article 19",
        "failure_mode": "Cross-document comparative: single embedding captures one document's angle only",
    },

    # ── Type D: Negation / exception (frequently fails) ───────────────────────
    {
        "id": "D1",
        "type": "negation",
        "question": "Under what circumstances is a person NOT liable for murder even if they cause death?",
        "expected_section": "Section 76-106 (General Exceptions)",
        "failure_mode": "Negation: embedding of 'not liable for murder' retrieves murder punishment, not exceptions",
    },
    {
        "id": "D2",
        "type": "negation",
        "question": "Which categories of information are exempt from disclosure under the RTI Act?",
        "expected_section": "Section 8",
        "failure_mode": "Negation framing: may retrieve general RTI info instead of exemption section",
    },

    # ── Type E: Out-of-corpus / hallucination trap ─────────────────────────────
    {
        "id": "E1",
        "type": "out_of_corpus",
        "question": (
            "What are the penalties for insider trading under SEBI regulations "
            "and how do they interact with IPC provisions on cheating?"
        ),
        "expected_section": "N/A — SEBI regulations likely not in corpus",
        "failure_mode": "Out-of-corpus: Gemini hallucinates SEBI details from training data",
    },
    {
        "id": "E2",
        "type": "out_of_corpus",
        "question": (
            "What did the Supreme Court rule in the Kesavananda Bharati case "
            "regarding the basic structure doctrine?"
        ),
        "expected_section": "N/A — case law likely not in corpus",
        "failure_mode": "Out-of-corpus: famous case, Gemini will confidently hallucinate from memory",
    },
]


# ── Scoring heuristics ─────────────────────────────────────────────────────────

def score_response(response: RAGResponse, question_meta: dict) -> dict:
    """
    Lightweight automated scoring — does NOT call the LLM.
    Uses retrieval signal only (score distribution) to flag likely failures.

    Thresholds are heuristic — adjust after seeing your corpus's score distribution.
    """
    top_score = response.top_score
    avg_score = response.avg_score
    answer_len = len(response.answer)
    has_no_info = any(phrase in response.answer.lower() for phrase in [
        "do not contain sufficient",
        "not enough information",
        "cannot find",
        "no relevant",
        "i don't know",
    ])

    # Detect likely hallucination: high-confidence long answer but low retrieval scores
    likely_hallucination = (
        top_score < 0.60
        and answer_len > 300
        and not has_no_info
    )

    # Detect retrieval failure: very low scores across the board
    retrieval_failed = top_score < 0.50

    # Detect honest "I don't know" — correct behavior for out-of-corpus questions
    correctly_abstained = has_no_info and question_meta["type"] == "out_of_corpus"

    diagnosis = "ok"
    if question_meta["type"] == "out_of_corpus":
        diagnosis = "correct_abstain" if correctly_abstained else "hallucination"
    elif retrieval_failed:
        diagnosis = "retrieval_failure"
    elif likely_hallucination:
        diagnosis = "likely_hallucination"
    elif top_score < 0.70:
        diagnosis = "weak_retrieval"

    return {
        "id":                  question_meta["id"],
        "type":                question_meta["type"],
        "question":            response.question[:80] + "...",
        "top_score":           round(top_score, 3),
        "avg_score":           round(avg_score, 3),
        "answer_length":       answer_len,
        "latency_ms":          round(response.latency_ms),
        "has_no_info":         has_no_info,
        "likely_hallucination":likely_hallucination,
        "diagnosis":           diagnosis,
        "expected_failure":    question_meta["failure_mode"],
        "chunks_retrieved":    len(response.retrieved_chunks),
        "sources":             [
            {
                "source":  c.metadata.get("source"),
                "page":    c.metadata.get("page"),
                "section": c.metadata.get("section"),
                "score":   round(c.score, 3),
            }
            for c in response.retrieved_chunks
        ],
    }


# ── Report printer ─────────────────────────────────────────────────────────────

DIAGNOSIS_EMOJI = {
    "ok":                "✓",
    "correct_abstain":   "✓",
    "weak_retrieval":    "⚠",
    "retrieval_failure": "✗",
    "likely_hallucination": "✗",
    "hallucination":     "✗",
}

def print_report(scores: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("NAIVE RAG FAILURE ANALYSIS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for s in scores:
        icon = DIAGNOSIS_EMOJI.get(s["diagnosis"], "?")
        print(f"\n[{s['id']}] {icon} {s['type'].upper()}")
        print(f"  Q: {s['question']}")
        print(f"  Top score:  {s['top_score']}  |  Avg: {s['avg_score']}  |  Chunks: {s['chunks_retrieved']}")
        print(f"  Latency:    {s['latency_ms']}ms  |  Answer length: {s['answer_length']} chars")
        print(f"  Diagnosis:  {s['diagnosis']}")
        print(f"  Why:        {s['expected_failure']}")
        if s["sources"]:
            top_src = s["sources"][0]
            print(f"  Top source: {top_src['source']} | pg {top_src['page']} | {top_src['section']} | score {top_src['score']}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    diagnoses = [s["diagnosis"] for s in scores]
    ok_count           = diagnoses.count("ok") + diagnoses.count("correct_abstain")
    weak_count         = diagnoses.count("weak_retrieval")
    failure_count      = diagnoses.count("retrieval_failure")
    hallucination_count = sum(1 for d in diagnoses if "hallucination" in d)

    total = len(scores)
    print(f"  Total questions:    {total}")
    print(f"  OK / abstained:     {ok_count}  ({100*ok_count//total}%)")
    print(f"  Weak retrieval:     {weak_count}  ({100*weak_count//total}%)")
    print(f"  Retrieval failure:  {failure_count}  ({100*failure_count//total}%)")
    print(f"  Likely hallucination: {hallucination_count}  ({100*hallucination_count//total}%)")

    avg_top_score = sum(s["top_score"] for s in scores) / total
    avg_latency   = sum(s["latency_ms"] for s in scores) / total
    print(f"\n  Avg top retrieval score: {avg_top_score:.3f}")
    print(f"  Avg latency:             {avg_latency:.0f}ms")

    print("\n" + "=" * 70)
    print("WHAT THESE FAILURES TELL YOU:")
    print("  Multi-hop (B): → RAG-Fusion fixes with multi-query decomposition")
    print("  Comparative (C): → RAG-Fusion generates both-sides queries")
    print("  Negation (D): → CRAG grades chunks and retries with rewritten query")
    print("  Out-of-corpus (E): → CRAG falls back to web search, Self-RAG abstains")
    print("=" * 70 + "\n")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_failure_analysis(save_json: bool = True) -> list[dict]:
    logger.info(f"Running failure analysis on {len(TEST_QUESTIONS)} questions...")

    rag = NaiveRAG(top_k=5)
    scores = []

    for i, q_meta in enumerate(TEST_QUESTIONS, 1):
        logger.info(f"[{i}/{len(TEST_QUESTIONS)}] {q_meta['id']}: {q_meta['question'][:60]}...")
        try:
            response = rag.query(q_meta["question"])
            score = score_response(response, q_meta)
            scores.append(score)
        except Exception as e:
            logger.error(f"  Failed: {e}")
            scores.append({
                "id": q_meta["id"],
                "type": q_meta["type"],
                "question": q_meta["question"][:80],
                "diagnosis": "error",
                "error": str(e),
            })

        # Small delay between questions to respect rate limits
        if i < len(TEST_QUESTIONS):
            time.sleep(1.0)

    print_report(scores)

    if save_json:
        out_path = Path("data/failure_analysis.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(scores, f, indent=2)
        logger.info(f"Full report saved to {out_path}")

    return scores


if __name__ == "__main__":
    run_failure_analysis()
