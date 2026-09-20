"""message_embeddings + file_chunks embedding -> halfvec(2048) (nemotron-3-embed-1b)

2026-09-19 NIM EOL recovery: nvidia/nv-embedqa-e5-v5 (1024-d) went 410 Gone; the only
live NIM embedder is nvidia/nemotron-3-embed-1b, fixed at 2048-d. Plain pgvector
`vector` cannot HNSW-index more than 2000 dimensions, so both embedding columns move
to `halfvec(2048)` (pgvector 0.8.2, live, supports halfvec + halfvec_cosine_ops).

Existing 1024-d values cannot be cast to 2048-d — there is no numeric relationship
between the two spaces. Both columns are dropped and re-added as **nullable**
halfvec(2048) rather than deleting rows: `services/re_embed.py` /
`services/arq_worker.py::re_embed_batch_job` re-embeds by SELECTing the existing
FileChunk/MessageEmbedding rows (ordered + paginated by offset) and UPDATEing
`.embedding` in place — it does not recreate rows. Deleting the rows instead would
have left `message_embeddings` permanently empty for every pre-migration message
(nothing else re-inserts a MessageEmbedding for an old message). `message_embeddings
.embedding` was NOT NULL pre-migration; relaxing it to nullable is required to leave
the rows in place with an empty vector until re-embed backfills them — the ORM
(`models/chat.py`) is updated to match. `file_chunks.embedding` was already nullable.

`check_and_queue_re_embed()` (called on app startup) detects the `MODEL_EMBEDDING`
change and enqueues `re_embed_batch_job` for every row in both tables, so the NULLs
left by this migration are transient once the ARQ workers run.

Revision ID: 049
Revises: 048
Create Date: 2026-09-19
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import HALFVEC, Vector

revision      = "049"
down_revision = "048"
branch_labels = None
depends_on    = None

_OLD_DIM = 1024
_NEW_DIM = 2048


def upgrade():
    # Indexes are bound to the column and would be dropped implicitly by DROP
    # COLUMN, but drop them explicitly first for clarity/idempotency (mirrors the
    # explicit-drop-before-column-change pattern in migration 017).
    op.execute("DROP INDEX IF EXISTS ix_file_chunks_hnsw")
    # Unnamed index from migration 004 (`CREATE INDEX ON message_embeddings USING
    # hnsw (...)`) — Postgres auto-names a single-column unnamed index
    # "<table>_<column>_idx".
    op.execute("DROP INDEX IF EXISTS message_embeddings_embedding_idx")
    op.execute("DROP INDEX IF EXISTS ix_message_embeddings_hnsw")

    op.execute("ALTER TABLE file_chunks DROP COLUMN IF EXISTS embedding")
    op.add_column("file_chunks", sa.Column("embedding", HALFVEC(_NEW_DIM), nullable=True))

    op.execute("ALTER TABLE message_embeddings DROP COLUMN IF EXISTS embedding")
    op.add_column("message_embeddings", sa.Column("embedding", HALFVEC(_NEW_DIM), nullable=True))

    op.execute(
        "CREATE INDEX ix_file_chunks_hnsw "
        "ON file_chunks USING hnsw (embedding halfvec_cosine_ops) "
        "WHERE embedding IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX ix_message_embeddings_hnsw "
        "ON message_embeddings USING hnsw (embedding halfvec_cosine_ops) "
        "WHERE embedding IS NOT NULL"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_file_chunks_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_message_embeddings_hnsw")

    op.execute("ALTER TABLE file_chunks DROP COLUMN IF EXISTS embedding")
    op.add_column("file_chunks", sa.Column("embedding", Vector(_OLD_DIM), nullable=True))

    # Restored nullable, NOT the original NOT NULL: symmetric with upgrade — the
    # 2048-d values cannot be cast back to 1024-d either, so this is a drop+re-add
    # that relies on a subsequent re-embed to repopulate. Enforcing NOT NULL again
    # here would require either an empty table or a fabricated placeholder vector,
    # neither of which is safe; leaving it nullable matches how the column actually
    # behaves until the re-embed queue drains.
    op.execute("ALTER TABLE message_embeddings DROP COLUMN IF EXISTS embedding")
    op.add_column("message_embeddings", sa.Column("embedding", Vector(_OLD_DIM), nullable=True))

    op.execute(
        "CREATE INDEX ix_file_chunks_hnsw "
        "ON file_chunks USING hnsw (embedding vector_cosine_ops) "
        "WHERE embedding IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX ix_message_embeddings_hnsw "
        "ON message_embeddings USING hnsw (embedding vector_cosine_ops) "
        "WHERE embedding IS NOT NULL"
    )
