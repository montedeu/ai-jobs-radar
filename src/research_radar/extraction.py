"""Extraction: papers go to the model as one batch, the mentions come back into the database.

Submitting and collecting are separate steps, because a batch takes minutes to hours and
the process that sent it may be gone by then. Everything the second step needs is in the
database. The model is reached only through ModelClient; tests pass a fake one.
"""

import hashlib
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import psycopg
from pydantic import ValidationError

from research_radar.mentions import PaperMentions
from research_radar.normalize import normal_form
from research_radar.papers import abstract_hash
from research_radar.settings import (
    BATCH_INPUT_USD_PER_MTOK,
    BATCH_OUTPUT_USD_PER_MTOK,
    EXTRACTION_CEILING_USD,
    EXTRACTION_MAX_OUTPUT_TOKENS,
    EXTRACTION_MODEL,
)

# A placeholder: the wording and the stop-list are written in #6 and #7.
EXTRACTION_PROMPT = (
    "List the methods, models, tasks and datasets that the abstract mentions. "
    "Copy each spelling exactly as written. If the abstract spells out an abbreviation, "
    "give the full form as its expansion."
)


def prompt_hash() -> str:
    return hashlib.sha256(EXTRACTION_PROMPT.encode()).hexdigest()


@dataclass(frozen=True)
class ExtractionRequest:
    arxiv_id: str
    title: str
    abstract: str


@dataclass(frozen=True)
class ExtractionResult:
    arxiv_id: str
    # None: the model answered, but not in the schema. It is still billed, so the tokens count.
    mentions: PaperMentions | None
    input_tokens: int
    output_tokens: int


class ModelClient(Protocol):
    """The only place the model is called."""

    def count_tokens(self, requests: Sequence[ExtractionRequest]) -> int: ...

    def submit_batch(self, requests: Sequence[ExtractionRequest]) -> str: ...

    def batch_ended(self, batch_id: str) -> bool: ...

    def batch_results(self, batch_id: str) -> Iterator[ExtractionResult]: ...


class BudgetExceeded(Exception):
    """The batch, with what is already spent, would pass the ceiling."""


def start_run(
    connection: psycopg.Connection, *, model: str, prompt_hash: str, note: str = ""
) -> int:
    run_id = connection.execute(
        "INSERT INTO extraction_run (model, prompt_hash, note) VALUES (%s, %s, %s) RETURNING id",
        (model, prompt_hash, note),
    ).fetchone()[0]
    connection.commit()
    return run_id


def _usd(input_tokens: int, output_tokens: int) -> Decimal:
    return (
        Decimal(input_tokens) * BATCH_INPUT_USD_PER_MTOK
        + Decimal(output_tokens) * BATCH_OUTPUT_USD_PER_MTOK
    ) / 1_000_000


def spent_usd(connection: psycopg.Connection) -> Decimal:
    """Money spent by every run: finished papers, unusable answers, and batches still out.

    A batch not yet collected counts at its estimate, which is the most it can cost.
    """
    input_tokens, output_tokens, pending = connection.execute(
        """
        SELECT
            (SELECT coalesce(sum(input_tokens), 0) FROM paper_extraction)
                + (SELECT coalesce(sum(unusable_input_tokens), 0) FROM extraction_batch),
            (SELECT coalesce(sum(output_tokens), 0) FROM paper_extraction)
                + (SELECT coalesce(sum(unusable_output_tokens), 0) FROM extraction_batch),
            (SELECT coalesce(sum(estimated_cost_usd), 0) FROM extraction_batch
              WHERE status = 'submitted')
        """
    ).fetchone()
    return _usd(input_tokens, output_tokens) + pending


def submit(
    connection: psycopg.Connection,
    client: ModelClient,
    run_id: int,
    papers: Sequence[ExtractionRequest],
    *,
    ceiling_usd: Decimal = EXTRACTION_CEILING_USD,
) -> int | None:
    """Sends the papers the run has not done as one batch; returns the batch's id.

    Returns None when there is nothing to send. Raises BudgetExceeded, before anything is
    sent, when the batch's most expensive outcome plus what is spent passes the ceiling.
    """
    if not papers:
        return None
    # Two submits at once would each see the same spent money and both pass the ceiling.
    connection.execute("SELECT pg_advisory_xact_lock(hashtext('extraction_budget'))")
    todo = _not_yet_extracted(connection, run_id, papers)
    if not todo:
        connection.rollback()
        return None
    # The output is counted at the cap for every paper: the real size is not known before.
    estimate = _usd(client.count_tokens(todo), len(todo) * EXTRACTION_MAX_OUTPUT_TOKENS)
    if spent_usd(connection) + estimate > ceiling_usd:
        connection.rollback()
        raise BudgetExceeded(f"batch estimated at ${estimate} does not fit under ${ceiling_usd}")
    provider_batch_id = client.submit_batch(todo)
    # The batch is paid for from here on, so the rows that find it again go in at once.
    batch_id = connection.execute(
        """
        INSERT INTO extraction_batch (run_id, provider_batch_id, estimated_cost_usd)
        VALUES (%s, %s, %s) RETURNING id
        """,
        (run_id, provider_batch_id, estimate),
    ).fetchone()[0]
    with connection.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO extraction_batch_paper (batch_id, arxiv_id, abstract_hash)"
            " VALUES (%s, %s, %s)",
            [(batch_id, p.arxiv_id, abstract_hash(p.abstract)) for p in todo],
        )
    connection.commit()
    return batch_id


def _not_yet_extracted(
    connection: psycopg.Connection, run_id: int, papers: Sequence[ExtractionRequest]
) -> list[ExtractionRequest]:
    """Drops papers the run has done, or has out in a batch, with the same abstract.

    Only a changed abstract sends a paper back (docs/adr/0003): the hash is of the abstract
    alone, so a new title does not.
    """
    unique = {p.arxiv_id: p for p in reversed(papers)}  # the first of a repeated id wins
    ids = list(unique)
    taken = connection.execute(
        """
        SELECT arxiv_id, abstract_hash FROM paper_extraction
         WHERE run_id = %(run)s AND arxiv_id = ANY(%(ids)s)
        UNION
        SELECT bp.arxiv_id, bp.abstract_hash
          FROM extraction_batch_paper bp JOIN extraction_batch b ON b.id = bp.batch_id
         WHERE b.run_id = %(run)s AND b.status = 'submitted' AND bp.arxiv_id = ANY(%(ids)s)
        """,
        {"run": run_id, "ids": ids},
    ).fetchall()
    taken_set = set(taken)
    return [
        p
        for p in reversed(unique.values())
        if (p.arxiv_id, abstract_hash(p.abstract)) not in taken_set
    ]


def collect(connection: psycopg.Connection, client: ModelClient, batch_id: int) -> Counter:
    """Stores a finished batch's answers. Counts "extracted", "mentions", "pending", "failed".

    A batch not yet ended changes nothing. A batch already collected is left alone, so
    running this again after a crash is safe. A paper whose answer is missing or not in the
    schema gets no completion mark and counts as failed; the rest of the batch is kept.
    """
    try:
        return _collect(connection, client, batch_id)
    except BaseException:
        connection.rollback()
        raise


def _collect(connection: psycopg.Connection, client: ModelClient, batch_id: int) -> Counter:
    # The lock makes two collectors of one batch take turns instead of storing it twice.
    row = connection.execute(
        "SELECT run_id, provider_batch_id, status FROM extraction_batch WHERE id = %s FOR UPDATE",
        (batch_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"no extraction batch {batch_id}")
    run_id, provider_batch_id, status = row
    if status == "collected":
        connection.rollback()
        return Counter()
    # The hash of the text that was sent, not the paper's current one: the abstract may
    # have changed while the batch ran, and the mentions belong to the old text.
    sent = dict(
        connection.execute(
            "SELECT arxiv_id, abstract_hash FROM extraction_batch_paper WHERE batch_id = %s",
            (batch_id,),
        ).fetchall()
    )
    if not client.batch_ended(provider_batch_id):
        connection.rollback()
        return Counter(pending=len(sent))

    counter: Counter = Counter()
    seen: set[str] = set()
    unusable_input = unusable_output = 0
    for result in client.batch_results(provider_batch_id):
        if result.arxiv_id not in sent or result.arxiv_id in seen:
            continue
        seen.add(result.arxiv_id)
        if result.mentions is None:
            unusable_input += result.input_tokens
            unusable_output += result.output_tokens
            counter["failed"] += 1
            continue
        counter["extracted"] += 1
        counter["mentions"] += _store(connection, run_id, sent[result.arxiv_id], result)
    if missing := len(sent) - len(seen):
        counter["failed"] += missing
    connection.execute(
        """
        UPDATE extraction_batch
           SET status = 'collected', collected_at = now(),
               unusable_input_tokens = %s, unusable_output_tokens = %s
         WHERE id = %s
        """,
        (unusable_input, unusable_output, batch_id),
    )
    connection.commit()
    return +counter  # drops zero counts


def _store(
    connection: psycopg.Connection, run_id: int, sent_hash: str, result: ExtractionResult
) -> int:
    """Marks the paper done in the run and replaces its mentions; returns how many it stored."""
    # A paper extracted again in the same run (its abstract changed) keeps its row, and
    # the tokens add up: both batches were billed.
    connection.execute(
        """
        INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens, output_tokens)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (run_id, arxiv_id) DO UPDATE SET
            abstract_hash = excluded.abstract_hash,
            input_tokens = paper_extraction.input_tokens + excluded.input_tokens,
            output_tokens = paper_extraction.output_tokens + excluded.output_tokens,
            extracted_at = now()
        """,
        (run_id, result.arxiv_id, sent_hash, result.input_tokens, result.output_tokens),
    )
    connection.execute(
        "DELETE FROM mention WHERE run_id = %s AND arxiv_id = %s", (run_id, result.arxiv_id)
    )
    # The key of a mention row is its type and spelling; a repeat adds nothing, except an
    # expansion the first one lacked.
    rows: dict[tuple[str, str], str | None] = {}
    for mention in result.mentions.mentions:
        key = (mention.entity_type, mention.spelling)
        rows[key] = rows.get(key) or mention.expansion
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO mention (run_id, arxiv_id, entity_type, spelling, normal_form, expansion)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (run_id, result.arxiv_id, entity_type, spelling, normal_form(spelling), expansion)
                for (entity_type, spelling), expansion in rows.items()
            ],
        )
    return len(rows)


def request_params(request: ExtractionRequest) -> dict[str, Any]:
    """The Messages API parameters for one paper; a batch entry and a token count share them."""
    from anthropic import transform_schema

    return {
        "model": EXTRACTION_MODEL,
        "system": EXTRACTION_PROMPT,
        "messages": [
            {"role": "user", "content": f"Title: {request.title}\n\nAbstract: {request.abstract}"}
        ],
        # The model's answer is held to the schema; transform_schema is the SDK's way to
        # make a pydantic schema one the API accepts. Low effort: reading an abstract for
        # names needs little thinking, and thinking tokens are billed as output.
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": transform_schema(PaperMentions)},
        },
    }


def result_from_batch_entry(entry: Any) -> ExtractionResult | None:
    """One line of a batch's results. None for a request that was not answered.

    An errored, canceled or expired request was not billed and has no tokens to record.
    Its paper gets no completion mark and is counted failed by collect.
    """
    if entry.result.type != "succeeded":
        return None
    message = entry.result.message
    text = next((block.text for block in message.content if block.type == "text"), None)
    try:
        mentions = PaperMentions.model_validate_json(text) if text else None
    except ValidationError:
        mentions = None
    return ExtractionResult(
        entry.custom_id, mentions, message.usage.input_tokens, message.usage.output_tokens
    )


class AnthropicModelClient:
    """ModelClient over the Message Batches API. Tests never build it (see conftest.py)."""

    def __init__(self) -> None:
        import anthropic

        # Credentials come from the environment or the developer's login, never the repo.
        self._client = anthropic.Anthropic()

    def count_tokens(self, requests: Sequence[ExtractionRequest]) -> int:
        # The endpoint counts one request at a time and does not bill.
        return sum(
            self._client.messages.count_tokens(**request_params(r)).input_tokens for r in requests
        )

    def submit_batch(self, requests: Sequence[ExtractionRequest]) -> str:
        entries = [
            {
                "custom_id": r.arxiv_id,
                "params": {**request_params(r), "max_tokens": EXTRACTION_MAX_OUTPUT_TOKENS},
            }
            for r in requests
        ]
        return self._client.messages.batches.create(requests=entries).id

    def batch_ended(self, batch_id: str) -> bool:
        return self._client.messages.batches.retrieve(batch_id).processing_status == "ended"

    def batch_results(self, batch_id: str) -> Iterator[ExtractionResult]:
        for entry in self._client.messages.batches.results(batch_id):
            if (result := result_from_batch_entry(entry)) is not None:
                yield result
