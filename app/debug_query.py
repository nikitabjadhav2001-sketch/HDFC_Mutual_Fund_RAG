"""CLI debug helper: run a retrieval query and print scored chunks.

Usage: python -m app.debug_query "what is the refund policy?"
"""

import argparse
import asyncio
import sys

from app.config import get_settings
from app.core.embedding import EmbeddingError
from app.rag.retriever import Retriever


async def run(query: str, k: int, threshold: float | None) -> int:
    if not get_settings().openai_api_key:
        print("error: OPENAI_API_KEY is not configured", file=sys.stderr)
        return 1

    retriever = Retriever()
    try:
        result = await retriever.retrieve(query, k=k, score_threshold=threshold)
    except EmbeddingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"top_score={result.top_score:.4f} "
        f"threshold={threshold if threshold is not None else 'settings'} "
        f"abstain={not result.is_sufficient}"
    )
    if not result.chunks:
        print("no chunks retrieved")
        return 1

    for chunk in result.chunks:
        print(
            f"[{chunk.citation_index}] score={chunk.score:.4f} "
            f"chunk={chunk.chunk_id} doc={chunk.doc_id}"
        )
        section = chunk.metadata.get("section_path")
        pages = chunk.metadata.get("page_start")
        if section or pages:
            print(f"    section={section} page_start={pages}")
        preview = chunk.text.replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"    {preview}")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.debug_query",
        description="Debug a retrieval query against the vector store.",
    )
    parser.add_argument("query", help="query text to retrieve for")
    parser.add_argument("--k", type=int, default=8, help="number of chunks to retrieve")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="minimum dense similarity for the abstention gate "
        "(defaults to settings.score_threshold)",
    )
    args = parser.parse_args(argv)

    if not args.query.strip():
        print("error: query must not be empty", file=sys.stderr)
        return 2

    return asyncio.run(run(args.query, args.k, args.threshold))


if __name__ == "__main__":
    raise SystemExit(main())
