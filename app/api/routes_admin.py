import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.api.auth import require_admin_key
from app.config import get_settings
from app.core.ratelimit import rate_limit
from app.db import get_db
from app.ingestion.parsers import ParseError, detect_format
from app.ingestion.queue import enqueue_ingest
from app.models.tables import Chunk as ChunkRow
from app.models.tables import Document
from app.schemas import DocumentOut, DocumentStatusOut, UploadOut

router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(require_admin_key), Depends(rate_limit)],
    tags=["admin"],
)
logger = logging.getLogger(__name__)


@router.post("/documents", response_model=UploadOut, status_code=201)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadOut:
    max_bytes = get_settings().max_upload_bytes
    content_length = int(request.headers.get("content-length") or 0)
    if content_length > max_bytes:
        raise HTTPException(status_code=413, detail=f"file exceeds {max_bytes} byte limit")
    filename = Path(file.filename or "upload").name
    try:
        fmt = detect_format(filename)
    except ParseError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"file exceeds {max_bytes} byte limit")
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    digest = hashlib.sha256(data).hexdigest()

    doc = db.execute(
        select(Document).where(Document.content_hash == digest)
    ).scalar_one_or_none()
    created = doc is None

    if doc is not None and doc.status == "done":
        return UploadOut(document=_to_out(db, doc), created=False)

    if doc is None:
        doc_id = uuid.uuid4()
        doc = Document(
            id=doc_id,
            title=Path(filename).stem or filename,
            format=fmt,
            content_hash=digest,
            status="queued",
            source_uri=_save_upload(data, doc_id, fmt),
        )
        db.add(doc)
    else:
        doc.status = "queued"
        doc.error = None
        if not doc.source_uri or not Path(doc.source_uri).is_file():
            doc.source_uri = _save_upload(data, doc.id, fmt)
    db.commit()
    db.refresh(doc)

    try:
        await enqueue_ingest(str(doc.id))
    except Exception as exc:  # noqa: BLE001 - queue outage must not crash the API
        logger.warning("enqueue failed for document %s: %s", doc.id, exc)
        doc.status = "failed"
        doc.error = f"enqueue failed: {exc}"
        db.commit()
        raise HTTPException(
            status_code=503,
            detail="ingest queue unavailable; retry later or run `python -m app.ingest <file>`",
        ) from exc

    return UploadOut(document=_to_out(db, doc), created=created)


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentOut]:
    docs = db.execute(
        select(Document).order_by(Document.uploaded_at.desc())
    ).scalars().all()
    counts = _chunk_counts(db, [doc.id for doc in docs])
    return [_to_out(db, doc, counts.get(doc.id, 0)) for doc in docs]


@router.get("/documents/{document_id}/status", response_model=DocumentStatusOut)
def document_status(document_id: uuid.UUID, db: Session = Depends(get_db)) -> DocumentStatusOut:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    counts = _chunk_counts(db, [doc.id])
    return DocumentStatusOut(
        id=str(doc.id),
        status=doc.status,
        error=doc.error,
        chunk_count=counts.get(doc.id, 0),
    )


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)) -> Response:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    db.execute(delete(ChunkRow).where(ChunkRow.doc_id == doc.id))
    _remove_file(doc.source_uri)
    db.delete(doc)
    db.commit()
    return Response(status_code=204)


def _chunk_counts(db: Session, doc_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    if not doc_ids:
        return {}
    rows = db.execute(
        select(ChunkRow.doc_id, func.count())
        .where(ChunkRow.doc_id.in_(doc_ids))
        .group_by(ChunkRow.doc_id)
    ).all()
    return {row[0]: row[1] for row in rows}


def _to_out(db: Session, doc: Document, chunk_count: int | None = None) -> DocumentOut:
    if chunk_count is None:
        chunk_count = _chunk_counts(db, [doc.id]).get(doc.id, 0)
    return DocumentOut(
        id=str(doc.id),
        title=doc.title,
        format=doc.format,
        status=doc.status,
        error=doc.error,
        content_hash=doc.content_hash,
        chunk_count=chunk_count,
        uploaded_at=doc.uploaded_at.isoformat() if doc.uploaded_at else None,
    )


def _save_upload(data: bytes, document_id: uuid.UUID, fmt: str) -> str:
    storage = Path(get_settings().storage_dir)
    storage.mkdir(parents=True, exist_ok=True)
    path = storage / f"{document_id}.{fmt}"
    path.write_bytes(data)
    return str(path)


def _remove_file(source_uri: str | None) -> None:
    if not source_uri:
        return
    storage = Path(get_settings().storage_dir).resolve()
    try:
        path = Path(source_uri).resolve()
    except OSError:
        return
    if storage in path.parents and path.is_file():
        path.unlink(missing_ok=True)
