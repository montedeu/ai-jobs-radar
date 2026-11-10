"""Job run log

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE job_run (
            id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            job         text        NOT NULL,
            started_at  timestamptz NOT NULL DEFAULT now(),
            finished_at timestamptz,
            status      text        NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'succeeded', 'failed')),
            details     jsonb       NOT NULL DEFAULT '{}'
        );
        CREATE INDEX job_run_job_started_at ON job_run (job, started_at DESC);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE job_run")
