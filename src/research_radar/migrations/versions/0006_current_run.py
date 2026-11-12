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
        -- "full" is a reserved word in SQL: quote it wherever it is named.
        -- A full run is one meant to cover the whole processed history; only such a run
        -- can become current. Rows that exist now are pilots, hence the default.
        -- The current run is the one with the latest current_since: a switch is a new
        -- value, and the earlier ones stay as the history of switches.
        ALTER TABLE extraction_run
            ADD COLUMN "full"        boolean     NOT NULL DEFAULT false,
            ADD COLUMN current_since timestamptz;
        """
    )


def downgrade() -> None:
    op.execute('ALTER TABLE extraction_run DROP COLUMN current_since, DROP COLUMN "full";')
