"""
Vector store using Chroma (local) with free local embeddings (sentence-transformers).
No API key needed for indexing or retrieval — only chat/report generation uses OpenAI.
"""
from pathlib import Path
import os
import sys
import tempfile
import uuid

# Streamlit Cloud can ship an old sqlite3; Chroma needs >= 3.35
try:
    import sqlite3

    if sqlite3.sqlite_version_info < (3, 35, 0):
        import pysqlite3  # type: ignore[import-untyped]

        sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, LOCAL_EMBEDDING_MODEL

# Module-level cache so the model loads once
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return _model


def get_embedding(text: str) -> list[float]:
    """Single text to embedding (local, free)."""
    model = _get_model()
    return model.encode(text).tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed (local, free)."""
    if not texts:
        return []
    model = _get_model()
    return model.encode(texts).tolist()


class DocVectorStore:
    """Chroma-backed store for document chunks with local embeddings."""

    def __init__(self, persist_dir: Path = CHROMA_DIR, collection_name: str = "doc_rag"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = self._create_client()
        self.collection_name = collection_name
        self._collection = None

    def _create_client(self):
        """
        Build a Chroma client with cloud-safe fallbacks.

        Streamlit Cloud: use a writable temp dir (not EphemeralClient — Chroma 1.x
        raises "Could not connect to tenant" there). Index is session-scoped anyway.
        Local: persist under data/chroma_db.
        """
        settings = Settings(anonymized_telemetry=False)
        is_streamlit_cloud = bool(os.getenv("STREAMLIT_SHARING_MODE")) or "/mount/src" in str(
            Path.cwd()
        )

        if is_streamlit_cloud:
            cloud_dir = Path(tempfile.gettempdir()) / "docparse_chroma"
            cloud_dir.mkdir(parents=True, exist_ok=True)
            try:
                return chromadb.PersistentClient(
                    path=str(cloud_dir),
                    settings=settings,
                )
            except Exception:
                pass

        try:
            return chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=settings,
            )
        except Exception:
            pass

        try:
            from chromadb.api.client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        return chromadb.EphemeralClient(settings=settings)

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks for RAG"},
            )
        return self._collection

    def add_chunks(self, chunks: list[tuple[str, dict]]):
        """Add chunks with metadata. Embeds locally (free)."""
        if not chunks:
            return
        texts = [c[0] for c in chunks]
        metadatas = [c[1] for c in chunks]
        # Chroma expects str values in metadata
        for m in metadatas:
            for k, v in m.items():
                if not isinstance(v, (str, int, float, bool)):
                    m[k] = str(v)

        embeddings = get_embeddings_batch(texts)
        ids = [str(uuid.uuid4()) for _ in chunks]
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    def query(self, query_text: str, top_k: int = 8) -> list[tuple[str, dict]]:
        """Return top_k (document, metadata) for query."""
        q_embedding = get_embedding(query_text)
        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        return list(zip(docs, metas))

    def has_documents(self) -> bool:
        """True if the collection exists and has at least one document."""
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def clear(self):
        """Remove all documents in the collection (for re-indexing)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # Collection doesn't exist yet — nothing to clear
        self._collection = None
