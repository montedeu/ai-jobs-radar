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
        -- paid for and is not in the schema; 'errored' is a request the API rejected,
        -- 'expired' and 'canceled' ones it did not run. Only 'invalid' and 'errored' count
        -- toward giving up on a paper.
        ALTER TABLE extraction_batch_paper
            ADD COLUMN outcome text
            CHECK (outcome IN ('extracted', 'invalid', 'errored', 'expired', 'canceled', 'stale'));
        COMMENT ON COLUMN extraction_batch_paper.outcome IS
            'NULL for a batch not collected yet, and for batches collected before migration '
            '0006; those are not counted as attempts when a run gives up on a paper';

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
