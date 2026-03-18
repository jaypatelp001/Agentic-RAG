"""
retrieval/qdrant_store.py
────────────────────────────────────────────────────────────────────────────────
Qdrant vector DB client — index chunks and search by semantic similarity.

KEY CONCEPT: How vector search works inside Qdrant
───────────────────────────────────────────────────
Qdrant uses HNSW (Hierarchical Navigable Small World) — a graph-based
approximate nearest neighbor (ANN) algorithm.

Think of HNSW as a multi-layer map:
  - Bottom layer: all 768-dim vectors as nodes, connected to their nearest neighbors
  - Upper layers: progressively sparser "highway" graphs for fast traversal
  - At search time: enter at the top layer, navigate down to find approximate
    nearest neighbors in O(log n) time instead of O(n)

This means searching 1 million legal document chunks takes ~10ms, not 10 seconds.

COLLECTION STRUCTURE:
  Collection name: lexrag_legal_docs
  Vector config:   768 dimensions, cosine distance
  Payload (metadata per point):
    - source, act_name, page, section, chunk_id, chunk_index, ...

FILTERING:
  Qdrant supports payload filters during search — you can do:
    "find top 5 chunks SIMILAR to query AND where act_name = 'Indian Penal Code'"
  This is used in Adaptive RAG (Phase 5) to restrict retrieval to a specific act.
"""

import os
import uuid
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)

from chunker import Chunk
from embedder import EMBEDDING_DIM

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────────

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "lexrag_legal_docs")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
UPSERT_BATCH    = 200   # Points per upsert batch — balance memory vs throughput


# ── Result dataclass ───────────────────────────────────────────────────────────

from dataclasses import dataclass, field

@dataclass
class SearchResult:
    """A single retrieved chunk with its relevance score."""
    text: str
    score: float          # cosine similarity 0→1 (higher = more relevant)
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.text[:70].replace("\n", " ")
        return (
            f"SearchResult(score={self.score:.3f}, "
            f"section={self.metadata.get('section', 'N/A')}, "
            f"text='{preview}...')"
        )


# ── Qdrant store ───────────────────────────────────────────────────────────────

class QdrantStore:
    """
    Wraps the Qdrant client for LexRAG operations.

    Main operations:
      - create_collection():  Set up the vector index (once)
      - upsert_chunks():      Store chunks + their vectors
      - search():             Retrieve top-k similar chunks for a query vector
      - search_with_filter(): Retrieve with metadata constraints (Adaptive RAG)
      - delete_collection():  Clean up for re-indexing
    """

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection: str = COLLECTION_NAME,
    ):
        self.collection = collection
        self.client = QdrantClient(host=host, port=port)
        logger.info(f"Connected to Qdrant at {host}:{port} | collection: '{collection}'")

    # ── Setup ──────────────────────────────────────────────────────────────────

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection if it doesn't exist.

        recreate=True: Delete and recreate — useful when re-ingesting after
                       changing chunk size or embedding model.

        Distance.COSINE: We use cosine similarity because embeddings are
        high-dimensional unit vectors — cosine distance is the right metric.
        (Euclidean distance works too but cosine is convention for text embeddings.)
        """
        exists = self.client.collection_exists(self.collection)

        if exists and not recreate:
            info = self.client.get_collection(self.collection)
            count = info.points_count
            logger.info(f"Collection '{self.collection}' already exists ({count} points). Skipping creation.")
            return

        if exists and recreate:
            logger.warning(f"Deleting existing collection '{self.collection}'...")
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.success(f"Created collection '{self.collection}' (dim={EMBEDDING_DIM}, metric=cosine)")

    # ── Indexing ───────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """
        Store chunks + their embedding vectors in Qdrant.

        UPSERT = INSERT OR UPDATE. If a chunk_id already exists, it gets
        overwritten. This means you can safely re-run ingestion without
        duplicating data — idempotent indexing.

        Each Qdrant point has:
          id:      string UUID (derived from chunk_id for reproducibility)
          vector:  the 768-dim embedding
          payload: all chunk metadata (source, page, section, text, ...)
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunks ({len(chunks)}) and vectors ({len(vectors)}) must match.")

        points = []
        for chunk, vector in zip(chunks, vectors):
            # Use UUID5 from chunk_id for deterministic, reproducible IDs
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.metadata["chunk_id"]))
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk.text,        # store text in payload for retrieval
                    **chunk.metadata,          # all metadata fields
                }
            ))

        total_batches = (len(points) + UPSERT_BATCH - 1) // UPSERT_BATCH
        logger.info(f"Upserting {len(points)} points in {total_batches} batches...")

        for i in range(0, len(points), UPSERT_BATCH):
            batch = points[i : i + UPSERT_BATCH]
            self.client.upsert(collection_name=self.collection, points=batch)
            logger.debug(f"  Batch {i//UPSERT_BATCH + 1}/{total_batches} upserted ({len(batch)} points)")

        logger.success(f"Indexed {len(points)} chunks into Qdrant collection '{self.collection}'")

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Retrieve top-k most similar chunks to a query vector.

        score_threshold: Minimum cosine similarity to include a result.
          0.0 = return everything (default for baseline RAG)
          0.75 = only return high-confidence matches (use in CRAG)

        Returns SearchResult list sorted by score descending.
        """
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        raw = response.points

        results = []
        for hit in raw:
            payload = hit.payload or {}
            results.append(SearchResult(
                text=payload.pop("text", ""),   # text is stored in payload
                score=hit.score,
                metadata=payload,
            ))

        return results

    def search_with_filter(
        self,
        query_vector: list[float],
        top_k: int = 5,
        act_name: Optional[str] = None,
        section: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Filtered search — restrict results to a specific act or section.

        Used in Adaptive RAG when the agent determines the question is about
        a specific act (e.g. "Under RTI, what is the time limit for response?")
        and wants to restrict retrieval to RTI Act documents only.

        Qdrant payload filters are applied BEFORE the ANN search for efficiency —
        they don't slow down retrieval linearly.
        """
        conditions = []
        if act_name:
            conditions.append(FieldCondition(key="act_name", match=MatchValue(value=act_name)))
        if section:
            conditions.append(FieldCondition(key="section", match=MatchValue(value=section)))

        query_filter = Filter(must=conditions) if conditions else None

        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        raw = response.points

        results = []
        for hit in raw:
            payload = hit.payload or {}
            results.append(SearchResult(
                text=payload.pop("text", ""),
                score=hit.score,
                metadata=payload,
            ))

        logger.debug(f"Filtered search (act={act_name}, section={section}) → {len(results)} results")
        return results

    # ── Utils ──────────────────────────────────────────────────────────────────

    def collection_info(self) -> dict:
        """Return basic stats about the collection."""
        info = self.client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "points_count": info.points_count,
            "vector_dim": EMBEDDING_DIM,
        }

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection)
        logger.warning(f"Deleted collection '{self.collection}'")


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: python -m retrieval.qdrant_store
    Requires: Qdrant running locally (docker run -p 6333:6333 qdrant/qdrant)
    Expected output:
      Connected to Qdrant at localhost:6333
      Created collection 'lexrag_test' (dim=768, metric=cosine)
      Indexed 2 chunks into Qdrant
      Search results for 'murder punishment':
        SearchResult(score=0.891, section=Section 302, text='Section 302...')
        SearchResult(score=0.423, section=Section 420, text='Section 420...')
    """
    import numpy as np
    from chunker import Chunk

    store = QdrantStore(collection="lexrag_test")
    store.create_collection(recreate=True)

    # Fake 768-dim vectors (replace with real embeddings in production)
    def random_vec():
        v = np.random.randn(768).astype(float).tolist()
        norm = sum(x**2 for x in v) ** 0.5
        return [x / norm for x in v]

    chunks = [
        Chunk(text="Section 302. Punishment for murder.", metadata={"chunk_id":"c0","source":"ipc.pdf","act_name":"Indian Penal Code","page":1,"section":"Section 302","chunk_index":0,"chunk_size":36}),
        Chunk(text="Section 420. Cheating and dishonestly inducing delivery of property.", metadata={"chunk_id":"c1","source":"ipc.pdf","act_name":"Indian Penal Code","page":5,"section":"Section 420","chunk_index":0,"chunk_size":65}),
    ]
    vectors = [random_vec(), random_vec()]

    store.upsert_chunks(chunks, vectors)

    print(f"\nCollection info: {store.collection_info()}")
    print(f"\nSearch results (random query vector):")
    results = store.search(random_vec(), top_k=2)
    for r in results:
        print(f"  {r}")

    store.delete_collection()
