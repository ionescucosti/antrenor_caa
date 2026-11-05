"""Embedding providers. All return L2-normalised vectors so cosine similarity
can be computed uniformly by ChromaDB."""
from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from typing import Protocol

import httpx

from rokid_backend.core.text import normalize_text

log = logging.getLogger(__name__)
_RETRY_STATUS = (429, 500, 502, 503, 504)


class EmbeddingError(RuntimeError):
    pass


def quota_exhausted(response: httpx.Response) -> bool:
    """OpenAI answers 429 both for rate limits (worth a retry) and for an
    account without credit (`insufficient_quota`: retrying is pointless)."""
    if response.status_code != 429:
        return False
    try:
        error = (response.json() or {}).get("error") or {}
    except ValueError:
        return False
    return "quota" in str(error.get("type") or "") or "credit" in str(error.get("code") or "")


def retry_delay(response: httpx.Response | None, attempt: int, base: float = 0.5, cap: float = 3.0) -> float:
    """Backoff for a transient failure: the server's Retry-After when it is
    short enough, otherwise a growing delay; never longer than `cap` seconds
    (the answer flow cannot wait for a long rate-limit window)."""
    delay = base * attempt
    if response is not None:
        header = response.headers.get("retry-after", "")
        try:
            delay = max(delay, float(header))
        except ValueError:
            pass
    return min(cap, delay)


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
    """OpenAI (or compatible) /embeddings endpoint. text-embedding-3-small is
    the fast default; text-embedding-3-large separates Romanian paraphrases
    better (changing the model requires reindexing)."""

    _DIMENSIONS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536, "text-embedding-ada-002": 1536}

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 30.0, retries: int = 2) -> None:
        if not api_key:
            raise ValueError("ROKID_EMBEDDING_API_KEY is required for the openai embedding provider.")
        self.name = f"openai:{model}"
        self.dimension = self._DIMENSIONS.get(model, 1536)
        self.retries = retries   # extra attempts on timeouts, connection errors and HTTP 429/5xx
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                    headers={"Authorization": f"Bearer {api_key}"})
        self._model = model

    def _request(self, batch: list[str]) -> httpx.Response:
        failures = 0
        while True:
            try:
                response = self._client.post("/embeddings", json={"model": self._model, "input": batch})
            except httpx.HTTPError as exc:
                failures += 1
                if failures > self.retries:
                    raise
                log.warning("embeddings call failed with %s; retry %d/%d", type(exc).__name__, failures, self.retries)
                time.sleep(retry_delay(None, failures))
                continue
            if quota_exhausted(response):
                raise EmbeddingError("Contul OpenAI nu mai are credit (insufficient_quota); adauga credit in platform.openai.com > Billing.")
            if response.status_code in _RETRY_STATUS and failures < self.retries:
                failures += 1
                log.warning("embeddings endpoint answered HTTP %s; retry %d/%d", response.status_code, failures, self.retries)
                time.sleep(retry_delay(response, failures))
                continue
            if response.status_code >= 400:
                raise EmbeddingError(f"Serviciul de embeddings a raspuns HTTP {response.status_code}.")
            return response

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = [t[:8000] for t in texts[start:start + 64]]
            data = sorted(self._request(batch).json()["data"], key=lambda item: item["index"])
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
