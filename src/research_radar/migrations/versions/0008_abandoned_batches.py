"""A batch status for a batch given up on

Revision ID: 0008
Revises: 0007
"""

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        -- 'abandoned': a batch left in 'sending' that a person gave up on. Nobody knows
        -- whether it reached the API, so its estimate stays in the spent total; its papers
        -- are no longer in flight, and collect skips it.
        ALTER TABLE extraction_batch DROP CONSTRAINT extraction_batch_status_check;
        ALTER TABLE extraction_batch ADD CONSTRAINT extraction_batch_status_check
            CHECK (status IN ('sending', 'submitted', 'collected', 'rejected', 'abandoned'));
        """
    )


def downgrade() -> None:
    op.execute(
        """
        -- A batch given up on goes back to 'sending', where it was left.
        UPDATE extraction_batch SET status = 'sending' WHERE status = 'abandoned';
        ALTER TABLE extraction_batch DROP CONSTRAINT extraction_batch_status_check;
        ALTER TABLE extraction_batch ADD CONSTRAINT extraction_batch_status_check
            CHECK (status IN ('sending', 'submitted', 'collected', 'rejected'));
        """
    )
