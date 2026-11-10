"""A job run can be skipped

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE job_run DROP CONSTRAINT job_run_status_check;
        ALTER TABLE job_run ADD CONSTRAINT job_run_status_check
            CHECK (status IN ('running', 'succeeded', 'failed', 'skipped'));
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE job_run SET status = 'failed' WHERE status = 'skipped';
        ALTER TABLE job_run DROP CONSTRAINT job_run_status_check;
        ALTER TABLE job_run ADD CONSTRAINT job_run_status_check
            CHECK (status IN ('running', 'succeeded', 'failed'));
        """
    )
