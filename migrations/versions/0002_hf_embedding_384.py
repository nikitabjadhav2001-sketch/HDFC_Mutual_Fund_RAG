"""switch embeddings to HF all-MiniLM-L6-v2 (384-dim)

Drops all chunk vectors (vectors from the previous provider cannot be mixed
with 384-dim embeddings), narrows the column to vector(384), recreates the
HNSW index, and requeues documents so `python -m app.ingest <source_uri>`
(or a fresh upload) re-embeds them with the configured provider.

Revision ID: 0002_hf_embedding_384
Revises: 0001_initial
Create Date: 2026-10-08
"""
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0002_hf_embedding_384"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

OLD_EMBEDDING_DIM = 1536
NEW_EMBEDDING_DIM = 384


def upgrade() -> None:
    # 1. Old vectors are unusable under the new embedder; the table must be
    #    empty before narrowing the column (pgvector rejects cross-dim casts).
    op.execute("DELETE FROM chunks")

    # 2. Narrow the column and rebuild the HNSW index.
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.alter_column(
        "chunks",
        "embedding",
        type_=Vector(NEW_EMBEDDING_DIM),
        existing_type=Vector(OLD_EMBEDDING_DIM),
        existing_nullable=False,
    )
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )

    # 3. Requeue so documents are re-ingested (and re-embedded) on demand.
    op.execute("UPDATE documents SET status = 'queued', error = NULL WHERE status = 'done'")


def downgrade() -> None:
    op.execute("DELETE FROM chunks")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.alter_column(
        "chunks",
        "embedding",
        type_=Vector(OLD_EMBEDDING_DIM),
        existing_type=Vector(NEW_EMBEDDING_DIM),
        existing_nullable=False,
    )
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
