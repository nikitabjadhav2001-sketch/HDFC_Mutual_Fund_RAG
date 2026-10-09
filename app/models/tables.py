import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from app.db import Base

EMBEDDING_DIM = 384


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(512))
    source_uri: Mapped[str | None] = mapped_column(Text, default=None)
    content_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    format: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    error: Mapped[str | None] = mapped_column(Text, default=None)
    tags: Mapped[list | None] = mapped_column(JSON, default=None)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    doc_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(Vector(EMBEDDING_DIM))
    section_path: Mapped[str | None] = mapped_column(Text, default=None)
    page_start: Mapped[int | None] = mapped_column(Integer, default=None)
    page_end: Mapped[int | None] = mapped_column(Integer, default=None)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    ordinal: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str | None] = mapped_column(String(128), default=None)
    title: Mapped[str | None] = mapped_column(String(512), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conv_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSON, default=None)
    retrieved_chunk_ids: Mapped[list | None] = mapped_column(JSON, default=None)
    model: Mapped[str | None] = mapped_column(String(128), default=None)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    trace_id: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
