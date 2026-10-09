"""initial tables

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-07
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    # 1. Ensure pgvector extension exists before creating vector columns or HNSW indexes
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("format", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column(
            "doc_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("section_path", sa.Text(), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", sa.String(128), nullable=True),
        sa.Column("title", sa.String(512), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column(
            "conv_id",
            sa.Uuid(),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=True),
        sa.Column("retrieved_chunk_ids", sa.JSON(), nullable=True),
        sa.Column("model", sa.String(128), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_messages_conv_id", "messages", ["conv_id"])


def downgrade() -> None:
    op.drop_index("ix_messages_conv_id", table_name="messages")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.drop_index("ix_chunks_doc_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
