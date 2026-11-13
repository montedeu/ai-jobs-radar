"""The log of extraction quality measurements

Revision ID: 0011
Revises: 0010
"""

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        -- One row per measurement, never rewritten: the before/after figures of ADR 0005 are
        -- read from here. RESTRICT keeps a measured run from being deleted under its figures.
        CREATE TABLE quality_measurement (
            id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            measured_at       timestamptz NOT NULL DEFAULT now(),
            run_id            bigint NOT NULL REFERENCES extraction_run ON DELETE RESTRICT,
            dictionary_sha256 text,   -- null before the first dictionary load
            gold_sha256       text   NOT NULL,
            label             text   NOT NULL CHECK (label IN ('before', 'after')),
            metrics           jsonb  NOT NULL
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE quality_measurement;")
