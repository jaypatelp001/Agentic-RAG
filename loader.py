"""
ingestion/loader.py
────────────────────────────────────────────────────────────────────────────────
PDF → structured Document objects.

KEY CONCEPT: Why we keep metadata from the start
─────────────────────────────────────────────────
Every document object carries metadata (source filename, page number, act name).
This metadata is stored alongside the embedding in Qdrant.
When the agent retrieves a chunk later, it can say:
  "According to IPC Section 302 (page 47)..."
instead of just hallucinating a source.

Metadata is not optional in production RAG. It's what makes the system
trustworthy for legal use cases.
"""

import fitz  # PyMuPDF
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from loguru import logger


@dataclass
class Document:
    """A single page/section of text with its metadata."""
    text: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"Document(source={self.metadata.get('source')}, page={self.metadata.get('page')}, text='{preview}...')"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_act_name(filename: str) -> str:
    """
    Infer a human-readable act name from the filename.
    e.g. 'ipc_1860.pdf' → 'Indian Penal Code 1860'
    """
    mapping = {
        "ipc": "Indian Penal Code",
        "rti": "Right to Information Act",
        "constitution": "Constitution of India",
        "crpc": "Code of Criminal Procedure",
        "iea": "Indian Evidence Act",
        "mv": "Motor Vehicles Act",
        "pocso": "Protection of Children from Sexual Offences Act",
    }
    lower = filename.lower()
    for key, name in mapping.items():
        if key in lower:
            year_match = re.search(r"\d{4}", filename)
            year = year_match.group() if year_match else ""
            return f"{name} {year}".strip()
    # Fallback: prettify the filename itself
    return Path(filename).stem.replace("_", " ").title()


def _clean_text(text: str) -> str:
    """
    Remove common PDF artifacts:
    - Multiple blank lines → single newline
    - Soft hyphens at line breaks (word-wrap artifacts)
    - Page headers/footers that repeat (simple heuristic)
    """
    # Remove soft hyphens
    text = text.replace("\u00ad", "")
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are only digits (page numbers mid-text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


# ── Main loader ────────────────────────────────────────────────────────────────

class PDFLoader:
    """
    Loads one or more PDFs and returns a flat list of Document objects,
    one per page. Chunking is done separately in chunker.py.

    Why page-level documents?
    ─────────────────────────
    Keeping pages intact before chunking gives us accurate page numbers.
    If we chunked during loading, we'd lose track of which page a chunk
    came from — making source citation impossible.
    """

    def __init__(self, min_page_chars: int = 100):
        """
        min_page_chars: skip pages with fewer characters than this
                        (covers-of-pages, blank pages, pure image pages)
        """
        self.min_page_chars = min_page_chars

    def load_file(self, filepath: str | Path) -> list[Document]:
        """Load a single PDF file → list of page-level Documents."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"PDF not found: {filepath}")
        if filepath.suffix.lower() != ".pdf":
            raise ValueError(f"Expected .pdf file, got: {filepath.suffix}")

        act_name = _detect_act_name(filepath.name)
        logger.info(f"Loading '{act_name}' from {filepath.name}")

        documents = []
        doc = fitz.open(str(filepath))

        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = page.get_text("text")  # "text" mode = plain text, no HTML
            clean = _clean_text(raw_text)

            if len(clean) < self.min_page_chars:
                logger.debug(f"  Skipping page {page_num + 1} — too short ({len(clean)} chars)")
                continue

            documents.append(Document(
                text=clean,
                metadata={
                    "source":    filepath.name,
                    "act_name":  act_name,
                    "page":      page_num + 1,      # 1-indexed for humans
                    "filepath":  str(filepath),
                    "doc_type":  "legal_act",
                    "char_count": len(clean),
                }
            ))

        doc.close()
        logger.success(f"  Loaded {len(documents)} pages from '{act_name}'")
        return documents

    def load_directory(self, directory: str | Path) -> list[Document]:
        """Load all PDFs in a directory recursively."""
        directory = Path(directory)
        pdf_files = sorted(directory.rglob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []

        logger.info(f"Found {len(pdf_files)} PDFs in {directory}")
        all_documents = []

        for pdf_path in pdf_files:
            try:
                docs = self.load_file(pdf_path)
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {pdf_path.name}: {e}")
                continue

        logger.success(f"Total: {len(all_documents)} pages loaded from {len(pdf_files)} files")
        return all_documents


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: python -m ingestion.loader
    Place any PDF in data/raw/ to test.
    Expected output:
      Loading 'Indian Penal Code 1860' from ipc_1860.pdf
      Loaded 312 pages from 'Indian Penal Code 1860'
      Total: 312 pages loaded from 1 files
      Sample: Document(source=ipc_1860.pdf, page=1, text='THE INDIAN PENAL CODE...')
    """
    loader = PDFLoader()
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    docs = loader.load_directory(raw_dir)

    if docs:
        print(f"\nSample document:\n{docs[0]}")
        print(f"\nMetadata keys: {list(docs[0].metadata.keys())}")
    else:
        print("No PDFs found. Add PDFs to data/raw/ and re-run.")