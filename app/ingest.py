"""CLI fallback: ingest a document synchronously (no worker required).

Usage: python -m app.ingest path/to/file.pdf
"""

import argparse
import hashlib
import sys
from pathlib import Path

from sqlalchemy import select

from app.core.embedding import build_embedder
from app.db import SessionLocal
from app.ingestion.parsers import ParseError, detect_format
from app.ingestion.pipeline import ingest
from app.models.tables import Document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest",
        description="Ingest a document synchronously (sync path for tests).",
    )
    parser.add_argument("path", type=Path, help="file to ingest (txt, md, pdf, docx)")
    parser.add_argument("--title", default=None, help="override the document title")
    args = parser.parse_args(argv)

    if not args.path.is_file():
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 2
    try:
        fmt = detect_format(args.path.name)
    except ParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    digest = hashlib.sha256(args.path.read_bytes()).hexdigest()
    source_uri = str(args.path.resolve())

    db = SessionLocal()
    try:
        doc = db.execute(
            select(Document).where(Document.content_hash == digest)
        ).scalar_one_or_none()
        if doc is not None and doc.status == "done":
            print(f"unchanged — already ingested as {doc.id} (done)")
            return 0
        if doc is None:
            doc = Document(
                title=args.title or args.path.stem,
                format=fmt,
                content_hash=digest,
                status="queued",
                source_uri=source_uri,
            )
            db.add(doc)
        else:
            doc.title = args.title or doc.title
            doc.format = fmt
            doc.status = "queued"
            doc.error = None
            doc.source_uri = source_uri
        db.commit()
        db.refresh(doc)

        embedder = build_embedder()
        doc = ingest(doc.id, db, embedder=embedder)
        detail = f" — {doc.error}" if doc.error else ""
        print(f"{doc.id} [{doc.status}]{detail}")
        return 0 if doc.status == "done" else 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
