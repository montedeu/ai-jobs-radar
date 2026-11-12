"""Extraction batches, and the full form of an abbreviation on a mention

Revision ID: 0005
Revises: 0004
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        -- Last column, so INSERT INTO mention VALUES (five values) keeps working.
        ALTER TABLE mention ADD COLUMN expansion text;

        -- One row per batch sent to the model. Submitting and collecting are separate
        -- steps that can be hours apart, so the state lives here, not in the process.
        -- The row is written as 'sending' before the batch leaves, so a batch that was
        -- paid for is never missing here; a failure in between leaves it 'sending', for
        -- a person to sort out. 'rejected' is a batch the API refused outright: nothing
        -- was created, so it costs nothing and its papers are due again. This table is
        -- also the ledger of spent money: it outlives the papers, and a run that has
        -- batches cannot be deleted.
        CREATE TABLE extraction_batch (
            id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            run_id             bigint      NOT NULL REFERENCES extraction_run ON DELETE RESTRICT,
            provider_batch_id  text        UNIQUE,   -- NULL while 'sending'
            status             text        NOT NULL DEFAULT 'sending'
                               CHECK (status IN ('sending', 'submitted', 'collected', 'rejected')),
            -- the most the batch could cost: counted input plus the output cap
            estimated_cost_usd numeric     NOT NULL,
            -- what every answer of the batch cost, unusable and outdated ones included
            actual_cost_usd    numeric,
            submitted_at       timestamptz NOT NULL DEFAULT now(),
            collected_at       timestamptz
        );

        -- The papers a batch carries and the hash of the text that was sent: the
        -- abstract may change while the batch runs, and paper_extraction must record
        -- the text the mentions came from.
        CREATE TABLE extraction_batch_paper (
            batch_id      bigint NOT NULL REFERENCES extraction_batch ON DELETE CASCADE,
            arxiv_id      text   NOT NULL REFERENCES paper ON DELETE CASCADE,
            abstract_hash text   NOT NULL,
            PRIMARY KEY (batch_id, arxiv_id)
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE extraction_batch_paper, extraction_batch;
        ALTER TABLE mention DROP COLUMN expansion;
        """
    )
