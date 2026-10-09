import logging

from app.config import get_settings
from app.core.vectorstore import ScoredChunk

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, score_threshold: float | None = None) -> None:
        settings = get_settings()
        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else getattr(settings, "score_threshold", 0.0)
        )

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int = 8,
    ) -> list[ScoredChunk]:
        """Filter and rank retrieved chunks based on relevance scores and score threshold."""
        if not chunks:
            return []

        # Filter out chunks below minimum score threshold
        filtered = [c for c in chunks if c.score >= self.score_threshold]

        # Sort descending by score
        sorted_chunks = sorted(filtered, key=lambda c: c.score, reverse=True)

        return sorted_chunks[:top_k]
