"""Full-text index over the title and abstract of a paper

Revision ID: 0010
Revises: 0009
"""

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        -- search_papers (#24) matches against exactly this expression; the planner uses the
        -- index only for the same text, so search.py repeats it character for character.
        CREATE INDEX paper_search ON paper
            USING gin (to_tsvector('english', title || ' ' || abstract));
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX paper_search;")
