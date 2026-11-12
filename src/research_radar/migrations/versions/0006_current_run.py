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

        -- What became of each paper of a collected batch. 'invalid' is an answer that was
        -- paid for and is not in the schema; 'unanswered' is a request the API errored,
        -- expired or canceled, which costs nothing. Only 'invalid' counts toward giving
        -- up on a paper. NULL: the batch is not collected, or was before this column.
        ALTER TABLE extraction_batch_paper
            ADD COLUMN outcome text
            CHECK (outcome IN ('extracted', 'invalid', 'unanswered', 'stale'));

        -- The check for giving up looks at a run's batches and at a paper's rows.
        CREATE INDEX extraction_batch_run ON extraction_batch (run_id);
        CREATE INDEX extraction_batch_paper_arxiv_id ON extraction_batch_paper (arxiv_id);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX extraction_batch_paper_arxiv_id, extraction_batch_run;
        ALTER TABLE extraction_batch_paper DROP COLUMN outcome;
        DROP TABLE extraction_run_scope;
        ALTER TABLE extraction_run DROP COLUMN current_since, DROP COLUMN is_full;
        """
    )
