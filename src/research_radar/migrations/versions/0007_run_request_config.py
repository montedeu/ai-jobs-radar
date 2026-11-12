"""The request configuration a run is pinned to

Revision ID: 0007
Revises: 0006
"""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        -- The settings of the request that are not in the prompt text: reasoning on or off
        -- and the output cap, as {"reasoning": bool, "max_output_tokens": int}. Their hash
        -- with the rest of the request is prompt_hash. The default exists only to let the
        -- column be added; an empty object is not a valid configuration, and submit refuses
        -- a run that has one.
        ALTER TABLE extraction_run
            ADD COLUMN request_config jsonb NOT NULL DEFAULT '{}';
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE extraction_run DROP COLUMN request_config;")
