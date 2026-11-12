"""The current extraction run

Revision ID: 0006
Revises: 0005
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        -- A full run is one meant to cover the whole processed history; only such a run
        -- can become current. Rows that exist now are pilots, hence the default.
        -- The current run is the one with the latest current_since: a switch is a new
        -- value, and the earlier ones stay as the history of switches.
        ALTER TABLE extraction_run
            ADD COLUMN is_full       boolean     NOT NULL DEFAULT false,
            ADD COLUMN current_since timestamptz;

        -- The papers a full run has to cover, written when the run is created: a paper
        -- that arrives later is not in it, so a run can finish while the history grows.
        CREATE TABLE extraction_run_scope (
            run_id   bigint NOT NULL REFERENCES extraction_run ON DELETE CASCADE,
            arxiv_id text   NOT NULL REFERENCES paper ON DELETE CASCADE,
            PRIMARY KEY (run_id, arxiv_id)
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE extraction_run_scope;
        ALTER TABLE extraction_run DROP COLUMN current_since, DROP COLUMN is_full;
        """
    )
