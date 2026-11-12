"""Extraction: papers go to the model as one batch, the mentions come back into the database.

Submitting and collecting are separate steps, because a batch takes minutes to hours and
the process that sent it may be gone by then. Everything the second step needs is in the
database. The model is reached only through ModelClient; tests pass a fake one.
"""

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import psycopg
from pydantic import ValidationError

from research_radar.mentions import PaperMentions
from research_radar.normalize import normal_form
from research_radar.papers import abstract_hash, text_hash
from research_radar.settings import (
    BATCH_PRICES_USD_PER_MTOK,
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
    return text_hash(EXTRACTION_PROMPT)


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


def _usd(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    price_in, price_out = BATCH_PRICES_USD_PER_MTOK[model]
    return (Decimal(input_tokens) * price_in + Decimal(output_tokens) * price_out) / 1_000_000


def _require_idle(connection: psycopg.Connection) -> None:
    # Both steps roll their own work back on failure; with the caller's transaction open
    # that would take the caller's work with it.
    if connection.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        raise ValueError("the connection has a transaction open; commit or roll it back first")


def spent_usd(connection: psycopg.Connection) -> Decimal:
    """Money spent by every run, from the batch ledger.

    A batch that is not collected counts at its estimate, which is the most it can cost,
    and so does one stuck in 'sending': nobody knows whether it left.
    """
    return connection.execute(
        "SELECT coalesce(sum(coalesce(actual_cost_usd, estimated_cost_usd)), 0)"
        " FROM extraction_batch"
    ).fetchone()[0]


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
    The batch is recorded as 'sending' before it is sent: a failure after that leaves the
    row, counted in the budget, instead of a paid batch nobody knows about.
    """
    _require_idle(connection)
    if not papers:
        return None
    with connection.transaction():
        model = _run_model(connection, run_id)
        todo = _not_yet_extracted(connection, run_id, papers)
    if not todo:
        return None
    # Counting is slow network work and takes no money; it stays outside the lock below.
    input_tokens = client.count_tokens(todo)
    # The output is counted at the cap for every paper: the real size is not known before.
    estimate = _usd(model, input_tokens, len(todo) * EXTRACTION_MAX_OUTPUT_TOKENS)
    with connection.transaction():
        # Two submits at once would each see the same spent money and both pass the ceiling.
        connection.execute("SELECT pg_advisory_xact_lock(hashtext('extraction_budget'))")
        # Another submit may have taken some of the papers while this one was counting.
        # The estimate then covers more than is sent, which is the safe side.
        todo = _not_yet_extracted(connection, run_id, todo)
        if not todo:
            return None
        if spent_usd(connection) + estimate > ceiling_usd:
            raise BudgetExceeded(
                f"batch estimated at ${estimate} does not fit under ${ceiling_usd}"
            )
        batch_id = connection.execute(
            "INSERT INTO extraction_batch (run_id, estimated_cost_usd) VALUES (%s, %s)"
            " RETURNING id",
            (run_id, estimate),
        ).fetchone()[0]
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO extraction_batch_paper (batch_id, arxiv_id, abstract_hash)"
                " VALUES (%s, %s, %s)",
                [(batch_id, p.arxiv_id, abstract_hash(p.abstract)) for p in todo],
            )
    provider_batch_id = client.submit_batch(todo)
    with connection.transaction():
        connection.execute(
            "UPDATE extraction_batch SET provider_batch_id = %s, status = 'submitted'"
            " WHERE id = %s",
            (provider_batch_id, batch_id),
        )
    return batch_id


def _run_model(connection: psycopg.Connection, run_id: int) -> str:
    row = connection.execute("SELECT model FROM extraction_run WHERE id = %s", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"no extraction run {run_id}")
    if row[0] not in BATCH_PRICES_USD_PER_MTOK:
        raise ValueError(f"no batch prices for model {row[0]!r}; add them to settings.py")
    return row[0]


def _not_yet_extracted(
    connection: psycopg.Connection, run_id: int, papers: Sequence[ExtractionRequest]
) -> list[ExtractionRequest]:
    """Drops papers the run has done, or has out in a batch, with the same abstract.

    Only a changed abstract sends a paper back (docs/adr/0003): the hash is of the abstract
    alone, so a new title does not. A paper that is not in the database is an error here,
    and not at the batch's INSERT, which comes after the client was called.
    """
    unique = {p.arxiv_id: p for p in reversed(papers)}  # the first of a repeated id wins
    ids = list(unique)
    known = {
        row[0]
        for row in connection.execute(
            "SELECT arxiv_id FROM paper WHERE arxiv_id = ANY(%s)", (ids,)
        ).fetchall()
    }
    if missing := sorted(set(ids) - known):
        raise ValueError(f"papers not in the database: {', '.join(missing)}")
    taken = connection.execute(
        """
        SELECT arxiv_id, abstract_hash FROM paper_extraction
         WHERE run_id = %(run)s AND arxiv_id = ANY(%(ids)s)
        UNION
        SELECT bp.arxiv_id, bp.abstract_hash
          FROM extraction_batch_paper bp JOIN extraction_batch b ON b.id = bp.batch_id
         WHERE b.run_id = %(run)s AND b.status IN ('sending', 'submitted')
           AND bp.arxiv_id = ANY(%(ids)s)
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
    """Stores a finished batch's answers.

    Counts "extracted", "mentions", "pending" (not ended yet), "unsent" (stuck in
    'sending', for a person to sort out), "failed" (no usable answer: the paper gets no
    completion mark) and "stale" (the paper's abstract changed since the batch was sent, so
    the answer is dropped and the paper stays due). A batch already collected is left
    alone, so running this again after a crash is safe.
    """
    _require_idle(connection)
    with connection.transaction():
        return _collect(connection, client, batch_id)


def _collect(connection: psycopg.Connection, client: ModelClient, batch_id: int) -> Counter:
    # The lock makes two collectors of one batch take turns instead of storing it twice.
    row = connection.execute(
        """
        SELECT b.run_id, b.provider_batch_id, b.status, r.model
          FROM extraction_batch b JOIN extraction_run r ON r.id = b.run_id
         WHERE b.id = %s FOR UPDATE OF b
        """,
        (batch_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"no extraction batch {batch_id}")
    run_id, provider_batch_id, status, model = row
    if status == "collected":
        return Counter()
    # The hash of the text that was sent: the abstract may have changed while the batch ran.
    sent = dict(
        connection.execute(
            "SELECT arxiv_id, abstract_hash FROM extraction_batch_paper WHERE batch_id = %s",
            (batch_id,),
        ).fetchall()
    )
    if status == "sending":
        return Counter(unsent=len(sent))
    if not client.batch_ended(provider_batch_id):
        return Counter(pending=len(sent))

    current = dict(
        connection.execute(
            "SELECT arxiv_id, abstract_hash FROM paper WHERE arxiv_id = ANY(%s)", (list(sent),)
        ).fetchall()
    )
    counter: Counter = Counter()
    seen: set[str] = set()
    input_tokens = output_tokens = 0
    for result in client.batch_results(provider_batch_id):
        # Every answer is billed, whatever becomes of it.
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        if result.arxiv_id not in sent or result.arxiv_id in seen:
            continue
        seen.add(result.arxiv_id)
        if result.mentions is None:
            counter["failed"] += 1
        elif current.get(result.arxiv_id) != sent[result.arxiv_id]:
            # The abstract changed after the send; a batch sent for the newer one may be
            # stored already, and this answer must not replace it.
            counter["stale"] += 1
        else:
            counter["extracted"] += 1
            counter["mentions"] += _store(connection, run_id, sent[result.arxiv_id], result)
    if missing := len(sent) - len(seen):
        counter["failed"] += missing
    connection.execute(
        """
        UPDATE extraction_batch
           SET status = 'collected', collected_at = now(), actual_cost_usd = %s
         WHERE id = %s
        """,
        (_usd(model, input_tokens, output_tokens), batch_id),
    )
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
    # expansion the first one lacked. A spelling of nothing but punctuation has an empty
    # normal form, which would make every such spelling one entity, so it is not stored.
    rows: dict[tuple[str, str], tuple[str, str | None]] = {}
    for mention in result.mentions.mentions:
        form = normal_form(mention.spelling)
        if not form:
            continue
        key = (mention.entity_type, mention.spelling)
        earlier = rows[key][1] if key in rows else None
        rows[key] = (form, earlier or mention.expansion)
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO mention (run_id, arxiv_id, entity_type, spelling, normal_form, expansion)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [
                (run_id, result.arxiv_id, entity_type, spelling, form, expansion)
                for (entity_type, spelling), (form, expansion) in rows.items()
            ],
        )
    return len(rows)


def request_params(request: ExtractionRequest) -> dict[str, Any]:
    """The Messages API parameters for one paper; a batch entry and a token count share them."""
    from anthropic import transform_schema

    return {
        "model": EXTRACTION_MODEL,
        "system": EXTRACTION_PROMPT,
        "messages": [{"role": "user", "content": _request_text(request)}],
        # The model's answer is held to the schema; transform_schema is the SDK's way to
        # make a pydantic schema one the API accepts. Low effort: reading an abstract for
        # names needs little thinking, and thinking tokens are billed as output.
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": transform_schema(PaperMentions)},
        },
    }


def _request_text(request: ExtractionRequest) -> str:
    return f"Title: {request.title}\n\nAbstract: {request.abstract}"


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


# About 100,000 tokens per counting call (a token is rarely under three characters of
# English text), well inside what the endpoint takes.
COUNT_CHARS_PER_CALL = 300_000


class AnthropicModelClient:
    """ModelClient over the Message Batches API. Tests never build it (see conftest.py)."""

    def __init__(self) -> None:
        import anthropic

        # The key comes from an environment variable; in Cloud Run Secret Manager sets it.
        self._client = anthropic.Anthropic()

    def _count(self, request: ExtractionRequest) -> int:
        return self._client.messages.count_tokens(**request_params(request)).input_tokens

    def count_tokens(self, requests: Sequence[ExtractionRequest]) -> int:
        """Counts the batch in a few calls instead of one per paper.

        What every request shares (system prompt, schema, framing) is counted once, on a
        request with no text, and multiplied by the number of requests. The papers' own
        text is counted glued together in parts. Each part carries the shared part once
        more, so the sum is an overestimate, which is the safe side for a ceiling. The
        endpoint counts and does not bill.
        """
        overhead = self._count(ExtractionRequest("", "", ""))
        total = len(requests) * overhead
        part: list[str] = []
        size = 0
        for request in requests:
            text = _request_text(request)
            if part and size + len(text) > COUNT_CHARS_PER_CALL:
                total += self._count_part(part)
                part, size = [], 0
            part.append(text)
            size += len(text)
        if part:
            total += self._count_part(part)
        return total

    def _count_part(self, texts: list[str]) -> int:
        # A part is one request whose message is the papers' texts one after another.
        return self._count(ExtractionRequest("", "", "\n\n".join(texts)))

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
