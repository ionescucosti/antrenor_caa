"""Thin wrapper over a persistent ChromaDB client. One pair of collections per
knowledge base: document chunks and verified questions."""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings


log = logging.getLogger(__name__)


def _dimension_mismatch(exc: Exception) -> bool:
    """ChromaDB fixes a collection's vector size at creation; after a change of
    embedding provider every old vector is unusable (even an emptied collection
    keeps its size)."""
    return "dimension" in str(exc).lower()


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict[str, Any]
    similarity: float


def _safe(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    return cleaned if len(cleaned) >= 3 else f"kb-{cleaned}"


class VectorStore:
    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        self._lock = threading.RLock()

    def _collection(self, kind: str, kb_id: str):
        return self._client.get_or_create_collection(
            f"{kind}__{_safe(kb_id)}", embedding_function=None, metadata={"hnsw:space": "cosine"})

    def upsert(self, kind: str, kb_id: str, ids: list[str], texts: list[str],
               embeddings: list[list[float]], metadatas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        with self._lock:
            col = self._collection(kind, kb_id)
            try:
                self._upsert_batches(col, ids, texts, embeddings, metadatas)
            except Exception as exc:  # noqa: BLE001 - chromadb's exception classes vary between versions
                if not _dimension_mismatch(exc):
                    raise
                log.warning("collection %s holds vectors of another embedding size: rebuilding it; "
                            "other documents/datasets of this knowledge base must be reindexed", col.name)
                self._client.delete_collection(col.name)
                self._upsert_batches(self._collection(kind, kb_id), ids, texts, embeddings, metadatas)

    @staticmethod
    def _upsert_batches(col, ids, texts, embeddings, metadatas) -> None:
        for start in range(0, len(ids), 500):
            sl = slice(start, start + 500)
            col.upsert(ids=ids[sl], documents=texts[sl], embeddings=embeddings[sl], metadatas=metadatas[sl])

    def delete_where(self, kind: str, kb_id: str, where: dict[str, Any]) -> None:
        with self._lock:
            col = self._collection(kind, kb_id)
            try:
                col.delete(where=where)
            except Exception:
                existing = col.get(where=where, include=[])
                if existing["ids"]:
                    col.delete(ids=existing["ids"])

    def query(self, kind: str, kb_id: str, embedding: list[float], k: int,
              where: dict[str, Any] | None = None) -> list[Hit]:
        with self._lock:
            col = self._collection(kind, kb_id)
            if col.count() == 0:
                return []
            try:
                result = col.query(query_embeddings=[embedding], n_results=min(k, col.count()), where=where,
                                   include=["documents", "metadatas", "distances"])
            except Exception as exc:  # noqa: BLE001
                if not _dimension_mismatch(exc):
                    raise
                log.warning("collection %s was indexed with another embedding provider; reindex it (no semantic hits until then)", col.name)
                return []
        hits = []
        for id_, doc, meta, dist in zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]):
            hits.append(Hit(id=id_, text=doc or "", metadata=dict(meta or {}), similarity=round(1.0 - float(dist), 4)))
        return hits

    def count(self, kind: str, kb_id: str) -> int:
        with self._lock:
            return self._collection(kind, kb_id).count()

    def drop_knowledge_base(self, kb_id: str) -> None:
        with self._lock:
            for kind in ("chunks", "questions"):
                try:
                    self._client.delete_collection(f"{kind}__{_safe(kb_id)}")
                except Exception:
                    pass

    def heartbeat(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False
