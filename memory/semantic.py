from __future__ import annotations

import uuid
from datetime import datetime, timezone

from memory.models import SemanticMemoryHit


class SemanticMemory:
    """Fuzzy recall over free text: 'show me the nginx config I edited last
    month', 'where is my docker-compose file'. Backed by a local Chroma
    collection with sentence-transformers embeddings — everything runs
    on-device, nothing is sent to a third party for this store.

    Heavy imports (chromadb, sentence-transformers) are done lazily inside
    __init__ so importing this module elsewhere (e.g. for type hints) stays
    cheap, and so the rest of the app still works if these optional deps
    aren't installed yet.
    """

    def __init__(
        self,
        chroma_path: str,
        collection_name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        import chromadb
        from chromadb.utils import embedding_functions

        self._client = chromadb.PersistentClient(path=chroma_path)
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=self._embed_fn
        )

    def add(self, text: str, metadata: dict | None = None) -> str:
        doc_id = str(uuid.uuid4())
        meta = dict(metadata or {})
        meta.setdefault("indexed_at", datetime.now(timezone.utc).isoformat())
        self._collection.add(documents=[text], metadatas=[meta], ids=[doc_id])
        return doc_id

    def search(self, query: str, n_results: int = 5) -> list[SemanticMemoryHit]:
        if self._collection.count() == 0:
            return []
        n_results = min(n_results, self._collection.count())
        result = self._collection.query(query_texts=[query], n_results=n_results)
        hits: list[SemanticMemoryHit] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, distances):
            score = 1.0 / (1.0 + dist)
            hits.append(SemanticMemoryHit(text=doc, metadata=meta, score=score))
        return hits

    def delete(self, doc_id: str) -> None:
        self._collection.delete(ids=[doc_id])

    def count(self) -> int:
        return self._collection.count()