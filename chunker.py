"""
ingestion/chunker.py
────────────────────────────────────────────────────────────────────────────────
Page-level Documents → smaller overlapping chunks ready for embedding.

KEY CONCEPT: Chunking strategy is the #1 RAG tuning lever
──────────────────────────────────────────────────────────
Bad chunking = bad retrieval = bad answers, no matter how good your LLM is.

Three chunking strategies exist:
  1. Fixed-size       — Split every N characters. Simple. Breaks mid-sentence.
  2. Sentence-based   — Split on sentence boundaries. Better. Misses legal structure.
  3. Recursive        — Try paragraph splits first, fall back to sentence, then char.
                        This is what we use. It respects legal document structure.

For legal docs specifically:
  - Sections (e.g. "Section 302") should stay together when possible
  - Overlap is critical: legal reasoning often spans boundaries
  - Chunk metadata must carry the section reference if detectable

CHUNK SIZE RATIONALE (chunk_size=800, overlap=150):
  - Gemini text-embedding-004 handles up to 2048 tokens
  - 800 chars ≈ 200 tokens — gives good semantic density without dilution
  - 150 char overlap ≈ 1-2 sentences — preserves cross-boundary context
  - In practice: 1 IPC section ≈ 2-4 chunks
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from loader import Document


# ── Chunk dataclass ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A text chunk ready to be embedded and stored in Qdrant.
    Inherits metadata from parent Document and adds chunk-specific fields.
    """
    text: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(id={self.metadata.get('chunk_id')}, "
            f"source={self.metadata.get('source')}, "
            f"page={self.metadata.get('page')}, "
            f"section='{self.metadata.get('section', 'N/A')}', "
            f"text='{preview}...')"
        )


# ── Section detector ───────────────────────────────────────────────────────────

_SECTION_PATTERNS = [
    # IPC-style: "Section 302", "Sec. 144", "S. 420"
    r"\bSection\s+\d+[A-Z]?\b",
    r"\bSec\.\s*\d+[A-Z]?\b",
    # Article-style: "Article 19", "Art. 21"
    r"\bArticle\s+\d+[A-Z]?\b",
    r"\bArt\.\s*\d+[A-Z]?\b",
    # Rule-style: "Rule 3", "Clause 4(b)"
    r"\bRule\s+\d+\b",
    r"\bClause\s+\d+[A-Za-z()]*",
]

_SECTION_REGEX = re.compile("|".join(_SECTION_PATTERNS), re.IGNORECASE)


def _detect_section(text: str) -> Optional[str]:
    """
    Extract the first section/article reference found in a text chunk.
    Used to enrich chunk metadata for structured filtering later.

    e.g. "Section 302 of the IPC provides that..." → "Section 302"
    """
    match = _SECTION_REGEX.search(text)
    return match.group().strip() if match else None


# ── Splitter logic ─────────────────────────────────────────────────────────────

def _recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Recursive character text splitter.

    Tries to split on these separators in order:
      1. Double newline  (paragraph boundary — preferred)
      2. Single newline  (line boundary)
      3. Period+space    (sentence boundary)
      4. Space           (word boundary — last resort)
      5. Empty string    (hard character split — absolute last resort)

    KEY INSIGHT: Why recursive > fixed-size
    ────────────────────────────────────────
    Fixed-size splitting can cut a sentence like:
      "...the accused is liable to punishment under Section" | "302 of the IPC..."
    Now "Section" and "302" are in different chunks. A query for "Section 302"
    won't retrieve either chunk reliably.

    Recursive splitting respects paragraph breaks first, so legal sections
    almost always stay intact.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps:
            # Absolute last resort: hard character split
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

        sep = seps[0]
        parts = text.split(sep) if sep else list(text)

        chunks = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single part is too large, recurse with next separator
                if len(part) > chunk_size:
                    sub_chunks = _split(part, seps[1:])
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1] if sub_chunks else ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    raw_chunks = _split(text, separators)

    # Apply overlap: each chunk starts with the last `chunk_overlap` chars
    # of the previous chunk — this preserves cross-boundary context.
    if len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        tail = overlapped[-1][-chunk_overlap:] if chunk_overlap > 0 else ""
        overlapped.append(tail + raw_chunks[i])

    return overlapped


# ── Main chunker ───────────────────────────────────────────────────────────────

class LegalChunker:
    """
    Converts page-level Documents into overlapping Chunks.

    Design decisions:
    ─────────────────
    - chunk_size=800:    ~200 tokens. Sweet spot for legal dense text.
    - chunk_overlap=150: ~1-2 sentences. Preserves cross-boundary reasoning.
    - min_chunk_chars=50: Discard noise chunks (headers, page refs).
    - Section detection: Adds 'section' field to metadata for SQL filtering later.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_chars: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a single Document into Chunks."""
        raw_chunks = _recursive_split(
            document.text,
            self.chunk_size,
            self.chunk_overlap,
        )

        chunks = []
        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if len(text) < self.min_chunk_chars:
                continue  # discard noise

            section = _detect_section(text)

            chunk_id = (
                f"{document.metadata.get('source', 'unknown')}"
                f"_p{document.metadata.get('page', 0)}"
                f"_c{i}"
            )

            chunks.append(Chunk(
                text=text,
                metadata={
                    **document.metadata,         # inherit all parent metadata
                    "chunk_id":    chunk_id,
                    "chunk_index": i,
                    "chunk_size":  len(text),
                    "section":     section,      # e.g. "Section 302" or None
                }
            ))

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk an entire list of Documents with progress logging."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.success(
            f"Chunked {len(documents)} pages → {len(all_chunks)} chunks "
            f"(avg {len(all_chunks)//max(len(documents),1)} chunks/page)"
        )
        return all_chunks


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: python -m ingestion.chunker
    Expected output:
      Chunked 1 pages → 3 chunks (avg 3 chunks/page)
      Chunk 0: Chunk(id=test_p1_c0, section='Section 302', text='...')
      Chunk 1: Chunk(id=test_p1_c1, section='Section 302', text='...')
    """
    from loader import Document

    sample = Document(
        text="""Section 302. Punishment for murder.
Whoever commits murder shall be punished with death, or imprisonment for life,
and shall also be liable to fine.

Section 303. Punishment for murder by life-convict.
Whoever, being under sentence of imprisonment for life, commits murder, shall
be punished with death.

Section 304. Punishment for culpable homicide not amounting to murder.
Whoever commits culpable homicide not amounting to murder shall be punished with
imprisonment for life, or imprisonment of either description for a term which
may extend to ten years, and shall also be liable to fine.""",
        metadata={"source": "test.pdf", "act_name": "Indian Penal Code", "page": 1}
    )

    chunker = LegalChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(sample)

    print(f"\nGenerated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")
        print(f"  Section detected: {chunk.metadata['section']}")
        print(f"  Text preview: {chunk.text[:100]}...\n")
