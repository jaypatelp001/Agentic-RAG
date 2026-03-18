"""
ingestion/pipeline.py
────────────────────────────────────────────────────────────────────────────────
Master ingestion script: Raw PDFs → fully indexed Qdrant collection.

Run this once before starting the agent:
  python -m ingestion.pipeline

Or with a custom data directory:
  python -m ingestion.pipeline --data-dir /path/to/pdfs --recreate

WHAT IT DOES:
  1. Load all PDFs from data/raw/ → page-level Documents
  2. Chunk each Document → overlapping Chunks with metadata
  3. Embed all chunks via Gemini text-embedding-004 (batched)
  4. Upsert chunks + vectors into Qdrant
  5. Populate SQLite metadata DB for Adaptive RAG (Phase 5)
  6. Print a summary report

IDEMPOTENT:
  Re-running this script is safe. Qdrant upsert overwrites existing points
  by chunk_id. SQLite INSERT OR REPLACE handles duplicate rows.
  Pass --recreate to wipe and rebuild the collection from scratch.
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from loader import PDFLoader
from chunker import LegalChunker
from embedder import GeminiEmbedder
from qdrant_store import QdrantStore

load_dotenv()


# ── SQLite: populate metadata DB for Adaptive RAG ─────────────────────────────

def init_sqlite(db_path: str) -> sqlite3.Connection:
    """
    Create the SQLite metadata database.

    WHY SQLITE IN ADDITION TO QDRANT?
    ───────────────────────────────────
    Qdrant is great for "which chunks are semantically similar to this query?"
    SQLite is great for "give me all sections from the RTI Act passed after 2005"

    In Adaptive RAG (Phase 5), the agent decides at runtime whether a question
    needs semantic search (Qdrant) or structured lookup (SQLite).

    Example structured query:
      "List all sections of the IPC that mention fine as punishment"
      → SQL: SELECT section, text FROM chunks WHERE act_name='Indian Penal Code'
               AND text LIKE '%fine%'
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id     TEXT PRIMARY KEY,
            source       TEXT,
            act_name     TEXT,
            page         INTEGER,
            section      TEXT,
            chunk_index  INTEGER,
            chunk_size   INTEGER,
            text         TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_act ON chunks(act_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_section ON chunks(section)")
    conn.commit()
    logger.info(f"SQLite DB ready at {db_path}")
    return conn


def populate_sqlite(conn: sqlite3.Connection, chunks) -> None:
    """Insert all chunks into SQLite. INSERT OR REPLACE = idempotent."""
    rows = [
        (
            c.metadata["chunk_id"],
            c.metadata.get("source"),
            c.metadata.get("act_name"),
            c.metadata.get("page"),
            c.metadata.get("section"),
            c.metadata.get("chunk_index"),
            c.metadata.get("chunk_size"),
            c.text,
        )
        for c in chunks
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    logger.success(f"Inserted {len(rows)} rows into SQLite")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    data_dir: str = "data/raw",
    recreate: bool = False,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> dict:
    """
    Full ingestion pipeline. Returns a summary dict.

    Steps:
      1. Load PDFs
      2. Chunk
      3. Embed (batched, with progress)
      4. Upsert to Qdrant
      5. Populate SQLite
    """
    start_time = time.time()
    summary = {}

    # ── 1. Load ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 / 5 — Loading PDFs")
    logger.info("=" * 60)

    loader = PDFLoader(min_page_chars=100)
    documents = loader.load_directory(data_dir)

    if not documents:
        logger.error(f"No documents loaded from {data_dir}. Aborting.")
        sys.exit(1)

    summary["documents_loaded"] = len(documents)

    # ── 2. Chunk ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 / 5 — Chunking")
    logger.info("=" * 60)
    logger.info(f"Strategy: Recursive | chunk_size={chunk_size} | overlap={chunk_overlap}")

    chunker = LegalChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(documents)

    summary["chunks_created"] = len(chunks)

    # ── 3. Embed ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 / 5 — Embedding (Gemini text-embedding-004)")
    logger.info("=" * 60)
    logger.info("NOTE: This step costs API credits. Each batch = 100 chunks.")

    embedder = GeminiEmbedder()
    vectors = embedder.embed_chunks(chunks)

    summary["vectors_created"] = len(vectors)

    # ── 4. Upsert to Qdrant ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 / 5 — Upserting to Qdrant")
    logger.info("=" * 60)

    store = QdrantStore()
    store.create_collection(recreate=recreate)
    store.upsert_chunks(chunks, vectors)

    info = store.collection_info()
    summary["qdrant_points"] = info["points_count"]

    # ── 5. SQLite ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 / 5 — Populating SQLite metadata DB")
    logger.info("=" * 60)

    db_path = os.getenv("SQLITE_DB_PATH", "./data/lexrag_metadata.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = init_sqlite(db_path)
    populate_sqlite(conn, chunks)
    conn.close()

    summary["sqlite_db"] = db_path

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    summary["elapsed_seconds"] = round(elapsed, 1)

    logger.info("\n" + "=" * 60)
    logger.success("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Pages loaded:       {summary['documents_loaded']}")
    logger.info(f"  Chunks created:     {summary['chunks_created']}")
    logger.info(f"  Vectors embedded:   {summary['vectors_created']}")
    logger.info(f"  Qdrant points:      {summary['qdrant_points']}")
    logger.info(f"  SQLite DB:          {summary['sqlite_db']}")
    logger.info(f"  Total time:         {elapsed:.1f}s")
    logger.info("=" * 60)
    logger.info("Next step: Run Phase 2 to build baseline RAG and observe its failures.")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LexRAG ingestion pipeline")
    parser.add_argument("--data-dir",  default="data/raw",  help="Directory containing PDF files")
    parser.add_argument("--recreate",  action="store_true", help="Wipe and recreate Qdrant collection")
    parser.add_argument("--chunk-size",    type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        recreate=args.recreate,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
