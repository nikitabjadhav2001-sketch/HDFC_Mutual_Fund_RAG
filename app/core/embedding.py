import logging
import math
import time
import zlib
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class EmbeddingError(Exception):
    """Raised when the embeddings API cannot be reached or returns an error."""


class EmbeddingProvider(Protocol):
    model: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        model: str,
        api_key: str,
        batch_size: int = 100,
        *,
        base_url: str | None = None,
        max_retries: int = 3,
        backoff: float = 0.5,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.batch_size = max(1, batch_size)
        self.max_retries = max_retries
        self.backoff = backoff
        self._client = client or httpx.Client(
            base_url=base_url or "https://api.openai.com/v1", timeout=30.0
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": batch}
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post("/embeddings", json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    body = response.json()
                    items = sorted(body["data"], key=lambda item: item["index"])
                    if len(items) != len(batch):
                        raise EmbeddingError(
                            f"embeddings API returned {len(items)} vectors for {len(batch)} inputs"
                        )
                    tokens = body.get("usage", {}).get("total_tokens")
                    logger.info(
                        "embedded %d texts (model=%s, total_tokens=%s)",
                        len(batch),
                        self.model,
                        tokens,
                    )
                    return [item["embedding"] for item in items]
                last_error = EmbeddingError(
                    f"embeddings API returned {response.status_code}: {response.text[:300]}"
                )
                if response.status_code not in RETRYABLE_STATUS:
                    break
            if attempt < self.max_retries and self.backoff > 0:
                time.sleep(self.backoff * (2**attempt))

        raise EmbeddingError(f"embedding failed after retries: {last_error}") from last_error


class HuggingFaceEmbeddingProvider:
    """HF Inference Providers feature-extraction for sentence-transformers models.

    Posts `{"inputs": [...], "normalize": true, "truncate": true}` to
    `{base_url}/models/{model}/pipeline/feature-extraction` and parses the
    `list[list[float]]` response. Requires `HF_TOKEN` for non-anonymous
    (non-rate-limited) access.
    """

    def __init__(
        self,
        model: str,
        token: str = "",
        batch_size: int = 32,
        *,
        base_url: str | None = None,
        max_retries: int = 3,
        backoff: float = 0.5,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.token = token
        self.batch_size = max(1, batch_size)
        self.max_retries = max_retries
        self.backoff = backoff
        self._client = client or httpx.Client(
            base_url=(base_url or "https://router.huggingface.co/hf-inference").rstrip("/"),
            timeout=60.0,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        path = f"/models/{self.model}/pipeline/feature-extraction"
        payload = {"inputs": batch, "normalize": True, "truncate": True}
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(path, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    vectors = self._parse(response.json(), len(batch))
                    logger.info(
                        "embedded %d texts (model=%s)", len(batch), self.model
                    )
                    return vectors
                last_error = EmbeddingError(
                    f"HF inference returned {response.status_code}: {response.text[:300]}"
                )
                if response.status_code not in RETRYABLE_STATUS:
                    break
            if attempt < self.max_retries and self.backoff > 0:
                time.sleep(self.backoff * (2**attempt))

        raise EmbeddingError(f"embedding failed after retries: {last_error}") from last_error

    @staticmethod
    def _parse(body: object, expected: int) -> list[list[float]]:
        # A single-string input yields a flat vector; batch input yields a list
        # of vectors. We always send batches, but tolerate a flattened response.
        if isinstance(body, list) and body and isinstance(body[0], (int, float)):
            body = [body]
        if not isinstance(body, list) or len(body) != expected:
            count = len(body) if isinstance(body, list) else "?"
            raise EmbeddingError(
                f"HF inference returned {count} vectors for {expected} inputs"
            )
        dims: set[int] = set()
        for vector in body:
            if (
                not isinstance(vector, list)
                or not vector
                or not all(isinstance(value, (int, float)) for value in vector)
            ):
                raise EmbeddingError("HF inference returned a malformed embedding vector")
            dims.add(len(vector))
        if len(dims) != 1:
            raise EmbeddingError(
                f"HF inference returned inconsistent vector dims: {sorted(dims)}"
            )
        return [[float(value) for value in vector] for vector in body]


class LocalEmbeddingProvider:
    """sentence-transformers running on-device — no API key, no usage costs.

    The model loads lazily on first use; the first call downloads it from the
    Hugging Face Hub (~90 MB for all-MiniLM-L6-v2) unless already cached.
    `normalize_embeddings=True` keeps vectors unit-length (cosine retrieval).
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        *,
        encoder: object | None = None,
    ) -> None:
        self.model = model
        self.batch_size = max(1, batch_size)
        self._encoder = encoder

    @property
    def encoder(self) -> object:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model)
        return self._encoder

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.encoder.encode(  # type: ignore[attr-defined]
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in vectors]


class HashEmbeddingProvider:
    """Deterministic offline embedder: L2-normalized hashed bag-of-words.

    Used for tests/CI and for environments without an embeddings API
    (`EMBEDDING_PROVIDER=hash`). Dense quality is coarse — sparse BM25
    carries lexical recall — but it is stable and free.
    """

    model = "hash-bow"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        from app.core.sparse import tokenize

        vector = [0.0] * self.dim
        for token in tokenize(text):
            vector[zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector


def build_embedder() -> EmbeddingProvider:
    """Construct the configured embedding provider (no hardcoded paths/models)."""
    from app.config import get_settings

    settings = get_settings()
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider(settings.embedding_model)
    if settings.embedding_provider == "huggingface":
        return HuggingFaceEmbeddingProvider(
            settings.embedding_model,
            settings.hf_token,
            base_url=settings.hf_base_url or None,
        )
    if settings.embedding_provider == "hash":
        return HashEmbeddingProvider(settings.embedding_dim)
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(
            settings.embedding_model,
            settings.openai_api_key,
            base_url=settings.openai_base_url or None,
        )
    raise ValueError(f"unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
