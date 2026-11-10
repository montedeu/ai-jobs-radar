"""What arXiv says about a paper's version, and how far each slice is collected

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE paper
            ADD COLUMN version          integer     NOT NULL,
            ADD COLUMN arxiv_updated_at timestamptz NOT NULL;

        -- One row per slice. Both marks move only after the work they describe is
        -- committed, so a failed run is repeated by the next one instead of skipped.
        CREATE TABLE collection_state (
            category            text PRIMARY KEY,
            -- every paper first submitted up to this day is stored
            backfilled_through  date,
            -- every paper or new version arXiv published up to this moment is stored
            updated_through     timestamptz
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE collection_state;
        ALTER TABLE paper DROP COLUMN version, DROP COLUMN arxiv_updated_at;
        """
    )
