"""Papers, extraction runs, mentions and the synonym dictionary

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE paper (
            arxiv_id           text PRIMARY KEY,   -- without the version suffix
            title              text        NOT NULL,
            abstract           text        NOT NULL,
            abstract_hash      text        NOT NULL,
            authors            jsonb       NOT NULL,
            primary_category   text        NOT NULL,
            first_version_date date        NOT NULL,
            collected_at       timestamptz NOT NULL DEFAULT now(),
            updated_at         timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX paper_first_version_date ON paper (first_version_date);

        -- Every category of the paper, the primary one included: a slice is "has the
        -- category", so slices overlap and a paper has one row per slice it belongs to.
        CREATE TABLE paper_category (
            arxiv_id text NOT NULL REFERENCES paper ON DELETE CASCADE,
            category text NOT NULL,
            PRIMARY KEY (arxiv_id, category)
        );
        CREATE INDEX paper_category_category ON paper_category (category, arxiv_id);

        CREATE TABLE extraction_run (
            id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            model       text        NOT NULL,
            prompt_hash text        NOT NULL,
            note        text        NOT NULL DEFAULT '',
            created_at  timestamptz NOT NULL DEFAULT now()
        );

        -- One row per paper a run has finished, whether or not it found anything: the
        -- denominator of a share is the papers here, not the papers collected.
        CREATE TABLE paper_extraction (
            run_id        bigint NOT NULL REFERENCES extraction_run ON DELETE CASCADE,
            arxiv_id      text   NOT NULL REFERENCES paper ON DELETE CASCADE,
            abstract_hash text   NOT NULL,   -- the text the mentions were extracted from
            input_tokens  integer NOT NULL,
            output_tokens integer NOT NULL,
            extracted_at  timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (run_id, arxiv_id)
        );

        CREATE TABLE mention (
            run_id      bigint NOT NULL,
            arxiv_id    text   NOT NULL,
            entity_type text   NOT NULL
                        CHECK (entity_type IN ('method', 'model', 'task', 'dataset')),
            spelling    text   NOT NULL,   -- as written in the abstract, never rewritten
            normal_form text   NOT NULL,
            PRIMARY KEY (run_id, arxiv_id, entity_type, spelling),
            FOREIGN KEY (run_id, arxiv_id) REFERENCES paper_extraction ON DELETE CASCADE
        );
        CREATE INDEX mention_normal_form ON mention (run_id, normal_form);

        CREATE TABLE synonym (
            normal_form    text PRIMARY KEY,
            canonical_name text NOT NULL
        );

        -- A paper's mentions under their canonical names: at most one row per paper,
        -- type and entity, however many spellings the abstract used. Trends read this,
        -- so a dictionary change shows up without touching the mentions.
        CREATE VIEW paper_entity AS
        SELECT DISTINCT
            m.run_id,
            m.arxiv_id,
            m.entity_type,
            coalesce(s.canonical_name, m.normal_form) AS canonical_name
        FROM mention m
        LEFT JOIN synonym s USING (normal_form);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP VIEW paper_entity;
        DROP TABLE synonym, mention, paper_extraction, extraction_run, paper_category, paper;
        """
    )
