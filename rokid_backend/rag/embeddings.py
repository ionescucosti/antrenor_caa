"""Embedding providers. All return L2-normalised vectors so cosine similarity
can be computed uniformly by ChromaDB."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

from rokid_backend.core.text import normalize_text


class EmbeddingProvider(Protocol):
    name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class HashEmbedding:
    """Deterministic, offline embedding built from hashed word and character
    n-grams. Good enough for tests, development and exact/near-exact matching;
    not a substitute for a real semantic model in production."""

    name = "hash-ngram-512"
    dimension = 512
    _token = re.compile(r"[a-z0-9]+")

    def _features(self, text: str) -> list[str]:
        norm = normalize_text(text)
        words = self._token.findall(norm)
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        padded = f" {norm} "
        feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dimension
            for feat in self._features(text):
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                weight = 1.0 if feat.startswith("c:") else 2.5
                vec[index] += sign * weight
            vectors.append(_normalise(vec))
        return vectors


class ChromaDefaultEmbedding:
    """ChromaDB's bundled all-MiniLM-L6-v2 (ONNX). Downloads the model once
    (~80 MB) into the Chroma cache; runs fully offline afterwards."""

    name = "chroma-all-MiniLM-L6-v2"
    dimension = 384

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._fn = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_normalise([float(v) for v in vec]) for vec in self._fn(texts)]


class OpenAIEmbedding:
    """OpenAI (or compatible) /embeddings endpoint."""

    dimension = 1536

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 30.0) -> None:
        if not api_key:
            raise ValueError("ROKID_EMBEDDING_API_KEY is required for the openai embedding provider.")
        self.name = f"openai:{model}"
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                    headers={"Authorization": f"Bearer {api_key}"})
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = [t[:8000] for t in texts[start:start + 64]]
            response = self._client.post("/embeddings", json={"model": self._model, "input": batch})
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            out.extend(_normalise(item["embedding"]) for item in data)
        return out


def build_embedding_provider(settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hash":
        return HashEmbedding()
    if settings.embedding_provider == "chroma_default":
        return ChromaDefaultEmbedding()
    if settings.embedding_provider == "openai":
        return OpenAIEmbedding(settings.embedding_api_key or settings.openai_api_key or settings.llm_api_key,
                               settings.embedding_model, settings.embedding_base_url)
    raise ValueError(f"Unknown embedding provider {settings.embedding_provider}")
