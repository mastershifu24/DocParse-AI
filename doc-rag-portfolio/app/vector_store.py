"""
Vector store using Chroma (local) with free local embeddings (sentence-transformers).
No API key needed for indexing or retrieval — only chat/report generation uses OpenAI.
"""
from pathlib import Path
import os
import uuid

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
        Some Chroma/runtime combinations on Streamlit Cloud can raise runtime
        errors while constructing PersistentClient.
        """
        settings = Settings(anonymized_telemetry=False)

        # Streamlit Cloud tends to be the environment where this crash appears.
        # Prefer ephemeral there unless explicitly overridden.
        force_ephemeral = os.getenv("DOC_PARSE_FORCE_EPHEMERAL", "").lower() in {"1", "true", "yes"}
        is_streamlit_cloud = bool(os.getenv("STREAMLIT_SHARING_MODE")) or "/mount/src" in str(Path.cwd())
        if force_ephemeral or is_streamlit_cloud:
            return chromadb.EphemeralClient(settings=settings)

        try:
            return chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=settings,
            )
        except Exception:
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
