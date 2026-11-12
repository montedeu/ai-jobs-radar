"""Versions of the synonym dictionary

Revision ID: 0009
Revises: 0008
"""

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        -- One row per change of the dictionary file's content: loaded_at is the date of the
        -- change, not of a deploy. The header (#24) and the measurement log (#12) are
        -- planned to read the last row.
        CREATE TABLE synonym_version (
            id          bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            loaded_at   timestamptz NOT NULL DEFAULT now(),
            file_sha256 text        NOT NULL,
            entries     integer     NOT NULL
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE synonym_version;")
