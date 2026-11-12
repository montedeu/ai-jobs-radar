"""Extraction: papers go to the model as one batch, the mentions come back into the database.

Submitting and collecting are separate steps, because a batch takes minutes to hours and
the process that sent it may be gone by then. Everything the second step needs is in the
database. The model is reached only through ModelClient; tests pass a fake one.
"""

import logging
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import psycopg
from pydantic import ValidationError

from research_radar.mentions import PaperMentions
from research_radar.normalize import normal_form
from research_radar.papers import text_hash
from research_radar.settings import (
    BATCH_PRICES_USD_PER_MTOK,
    CATEGORIES,
    EXTRACTION_CEILING_USD,
    EXTRACTION_MAX_OUTPUT_TOKENS,
    HISTORY_START,
)

log = logging.getLogger(__name__)

# A placeholder: the wording and the stop-list are written in #6 and #7.
EXTRACTION_PROMPT = (
    "List the methods, models, tasks and datasets that the abstract mentions. "
    "Copy each spelling exactly as written. If the abstract spells out an abbreviation, "
    "give the full form as its expansion."
)


# The Batches API takes a custom_id of letters, digits, underscore and hyphen, up to 64
# characters; an arXiv id has a dot. A new-style id has nothing else, so the swap is
# reversible. An id of another shape (an old "hep-th/9901001") cannot be sent; submit
# leaves such papers out.
_ARXIV_ID = re.compile(r"[0-9]{4}\.[0-9]{4,5}")


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


class BatchRejected(Exception):
    """The API refused to create the batch outright: nothing was created, nothing is billed.

    Any other failure of submit_batch (a timeout, a dropped connection, a 5xx) leaves it
    unknown whether the batch exists, and must propagate as it is.
    """


class ModelClient(Protocol):
    """The only place the model is called."""

    def count_tokens(self, requests: Sequence[ExtractionRequest], *, model: str) -> int: ...

    def submit_batch(self, requests: Sequence[ExtractionRequest], *, model: str) -> str:
        """Returns the provider's batch id. Raises BatchRejected for a refusal that
        certainly created nothing."""
        ...

    def batch_ended(self, batch_id: str) -> bool: ...

    def batch_results(self, batch_id: str) -> Iterator[ExtractionResult]: ...


# How many times submit counts tokens when abstracts change under it, before it gives up.
MAX_COUNTS = 2


class AbstractsKeepChanging(Exception):
    """Papers changed under every count of a submit; try again."""


class BudgetExceeded(Exception):
    """The batch, with what is already spent, would pass the ceiling."""


def start_run(
    connection: psycopg.Connection,
    *,
    model: str,
    prompt_hash: str,
    note: str = "",
    full: bool = False,
) -> int:
    """Creates a run. A full run is one meant to cover the whole history (see runs.py).

    A full run also records its scope in the same transaction: the papers of the slices
    since HISTORY_START that exist now, with an id that can be extracted at all. Papers
    that come later, or gain a slice later, are not in it; the daily job sends them to
    the current run.
    """
    with connection.transaction():
        run_id = connection.execute(
            "INSERT INTO extraction_run (model, prompt_hash, note, is_full)"
            " VALUES (%s, %s, %s, %s) RETURNING id",
            (model, prompt_hash, note, full),
        ).fetchone()[0]
        if full:
            connection.execute(
                """
                INSERT INTO extraction_run_scope (run_id, arxiv_id)
                SELECT %(run)s, p.arxiv_id FROM paper p
                 WHERE p.first_version_date >= %(start)s
                   AND p.arxiv_id ~ %(shape)s
                   AND EXISTS (SELECT 1 FROM paper_category c
                                WHERE c.arxiv_id = p.arxiv_id AND c.category = ANY(%(slices)s))
                """,
                {
                    "run": run_id,
                    "start": HISTORY_START,
                    "shape": f"^{_ARXIV_ID.pattern}$",
                    "slices": list(CATEGORIES),
                },
            )
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
    and so does one stuck in 'sending': nobody knows whether it left. A rejected batch was
    never created and counts for nothing.
    """
    return connection.execute(
        "SELECT coalesce(sum(coalesce(actual_cost_usd, estimated_cost_usd)), 0)"
        " FROM extraction_batch WHERE status <> 'rejected'"
    ).fetchone()[0]


def submit(
    connection: psycopg.Connection,
    client: ModelClient,
    run_id: int,
    arxiv_ids: Sequence[str],
    *,
    ceiling_usd: Decimal = EXTRACTION_CEILING_USD,
) -> int | None:
    """Sends the papers the run has not done as one batch; returns the batch's id.

    Title and abstract are read from the database here, so the hash recorded for what was
    sent is always the hash of what was sent. Returns None when there is nothing to send.
    Raises BudgetExceeded, before anything is sent, when the batch's most expensive outcome
    plus what is spent passes the ceiling. Ids of a shape that cannot be sent (see
    _ARXIV_ID) are left out of the batch and logged as a warning.

    The batch is recorded as 'sending' before it is sent: a failure after that leaves the
    row, counted in the budget, instead of a paid batch nobody knows about. Only a refusal
    that certainly created nothing (BatchRejected) takes the money back and frees the papers.
    """
    _require_idle(connection)
    arxiv_ids = _supported(arxiv_ids)
    if not arxiv_ids:
        return None
    with connection.transaction():
        model = _run_model(connection, run_id)
        todo = _due(connection, run_id, arxiv_ids)
    if not todo:
        return None
    for _attempt in range(MAX_COUNTS):
        # Counting is slow network work and takes no money; it stays outside the lock below.
        input_tokens = client.count_tokens([request for request, _ in todo], model=model)
        # The output is counted at the cap for every paper: the real size is not known before.
        estimate = _usd(model, input_tokens, len(todo) * EXTRACTION_MAX_OUTPUT_TOKENS)
        with connection.transaction():
            # Two submits at once would each see the same spent money and both pass the ceiling.
            connection.execute("SELECT pg_advisory_xact_lock(hashtext('extraction_budget'))")
            counted = {request.arxiv_id: request for request, _ in todo}
            # Another submit may have taken some of the papers while this one was counting;
            # then the estimate covers more than is sent, the safe side.
            todo = _due(connection, run_id, list(counted))
            if not todo:
                return None
            if any(counted[request.arxiv_id] != request for request, _ in todo):
                # The title or abstract changed since the count, so the estimate is of
                # another text.
                # Leaving the block frees the lock before the slow count is done again.
                continue
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
                    [(batch_id, request.arxiv_id, abstract) for request, abstract in todo],
                )
            break
    else:
        raise AbstractsKeepChanging(f"papers changed during each of {MAX_COUNTS} counts")
    try:
        provider_batch_id = client.submit_batch([request for request, _ in todo], model=model)
    except BatchRejected:
        with connection.transaction():
            connection.execute(
                "UPDATE extraction_batch SET status = 'rejected' WHERE id = %s", (batch_id,)
            )
        raise
    with connection.transaction():
        connection.execute(
            "UPDATE extraction_batch SET provider_batch_id = %s, status = 'submitted'"
            " WHERE id = %s",
            (provider_batch_id, batch_id),
        )
    return batch_id


def _supported(arxiv_ids: Sequence[str]) -> list[str]:
    """The ids of a shape that can be sent; the others are left out and logged."""
    unique = list(dict.fromkeys(arxiv_ids))
    supported = {i for i in unique if _ARXIV_ID.fullmatch(i)}
    if len(supported) < len(unique):
        unsupported = [i for i in unique if i not in supported]
        log.warning("not sent, arXiv id of an unsupported shape: %s", ", ".join(unsupported))
    return [i for i in unique if i in supported]


def _run_model(connection: psycopg.Connection, run_id: int) -> str:
    row = connection.execute("SELECT model FROM extraction_run WHERE id = %s", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"no extraction run {run_id}")
    if row[0] not in BATCH_PRICES_USD_PER_MTOK:
        raise ValueError(f"no batch prices for model {row[0]!r}; add them to settings.py")
    return row[0]


def _due(
    connection: psycopg.Connection, run_id: int, arxiv_ids: Sequence[str]
) -> list[tuple[ExtractionRequest, str]]:
    """The papers to send, with the hash of each one's abstract as stored.

    Drops papers the run has done, or has out in a batch, with the same abstract. Only a
    changed abstract sends a paper back (docs/adr/0003): the hash is of the abstract alone,
    so a new title does not. A paper that is not in the database is an error here, and not
    at the batch's INSERT, which comes after the client was called.
    """
    ids = list(dict.fromkeys(arxiv_ids))  # a repeated id counts once, in the order given
    stored = {
        row[0]: (ExtractionRequest(*row[:3]), row[3])
        for row in connection.execute(
            "SELECT arxiv_id, title, abstract, abstract_hash FROM paper WHERE arxiv_id = ANY(%s)",
            (ids,),
        ).fetchall()
    }
    if missing := [i for i in ids if i not in stored]:
        raise ValueError(f"papers not in the database: {', '.join(missing)}")
    taken = set(
        connection.execute(
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
    )
    return [stored[i] for i in ids if (i, stored[i][1]) not in taken]


def collect(connection: psycopg.Connection, client: ModelClient, batch_id: int) -> Counter:
    """Stores a finished batch's answers.

    Counts "extracted", "mentions", "pending" (not ended yet), "unsent" (stuck in
    'sending', for a person to sort out), "rejected" (the API refused the batch), "failed"
    (the paper's request got no usable answer, or none at all because it errored, expired
    or was canceled: no completion mark, so the paper is due again), "stale" (the paper's
    abstract changed since the batch was sent, so the answer is dropped and the paper stays
    due) and "incomplete" (the stream has no result at all for some of the batch's papers,
    which is a download that broke off: nothing is stored, the batch stays 'submitted' and
    the next call tries again). A batch already collected is left alone, so running this
    again after a crash is safe.

    The network is read first, with no transaction and no lock; the database is written in
    one short transaction after, so a slow download holds nothing.
    """
    _require_idle(connection)
    with connection.transaction():
        row = connection.execute(
            "SELECT status, provider_batch_id,"
            " (SELECT count(*) FROM extraction_batch_paper WHERE batch_id = b.id)"
            " FROM extraction_batch b WHERE id = %s",
            (batch_id,),
        ).fetchone()
    if row is None:
        raise LookupError(f"no extraction batch {batch_id}")
    status, provider_batch_id, papers = row
    if status == "collected":
        return Counter()
    if status in ("sending", "rejected"):
        return Counter({"unsent" if status == "sending" else "rejected": papers})
    if not client.batch_ended(provider_batch_id):
        return Counter(pending=papers)
    results = list(client.batch_results(provider_batch_id))
    with connection.transaction():
        return _store_batch(connection, batch_id, results)


def _store_batch(
    connection: psycopg.Connection, batch_id: int, results: list[ExtractionResult]
) -> Counter:
    # The lock makes two collectors of one batch take turns; the status is read again
    # because the other may have stored the batch while this one was downloading it.
    run_id, status, model = connection.execute(
        """
        SELECT b.run_id, b.status, r.model
          FROM extraction_batch b JOIN extraction_run r ON r.id = b.run_id
         WHERE b.id = %s FOR UPDATE OF b
        """,
        (batch_id,),
    ).fetchone()
    if status != "submitted":
        return Counter()
    # The hash of the text that was sent: the abstract may have changed while the batch ran.
    sent = dict(
        connection.execute(
            "SELECT arxiv_id, abstract_hash FROM extraction_batch_paper WHERE batch_id = %s",
            (batch_id,),
        ).fetchall()
    )
    answered = {result.arxiv_id for result in results} & set(sent)
    if len(answered) < len(sent):
        # The Batches API gives a line for every request, so a short stream is a download
        # that broke off. Storing part of it would mark the batch done and lose the rest.
        return Counter(incomplete=len(sent) - len(answered))
    # Shared locks keep an abstract from changing between this check and the commit. They
    # come after the check above, so a batch that is not stored holds nothing of the papers.
    current = dict(
        connection.execute(
            "SELECT arxiv_id, abstract_hash FROM paper WHERE arxiv_id = ANY(%s) FOR SHARE",
            (list(sent),),
        ).fetchall()
    )
    counter: Counter = Counter()
    seen: set[str] = set()
    input_tokens = output_tokens = 0
    for result in results:
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


def request_params(request: ExtractionRequest, model: str) -> dict[str, Any]:
    """The Messages API parameters for one paper; a batch entry and a token count share them."""
    from anthropic import transform_schema

    return {
        "model": model,
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


_REFUSED_BEFORE_CREATING = frozenset({400, 401, 403, 404, 409, 413, 422, 429})


def _custom_id(arxiv_id: str) -> str:
    if not _ARXIV_ID.fullmatch(arxiv_id):
        raise BatchRejected(f"cannot send arXiv id {arxiv_id!r}: not a new-style id")
    return arxiv_id.replace(".", "_")


def result_from_batch_entry(entry: Any) -> ExtractionResult:
    """One line of a batch's results, as a result.

    An errored, canceled or expired request was not billed: it becomes a result with no
    mentions and no tokens, so that it is in the stream and collect counts it failed. Left
    out, it would make the whole batch look cut off.
    """
    if entry.result.type != "succeeded":
        return ExtractionResult(entry.custom_id.replace("_", "."), None, 0, 0)
    message = entry.result.message
    text = next((block.text for block in message.content if block.type == "text"), None)
    try:
        mentions = PaperMentions.model_validate_json(text) if text else None
    except ValidationError:
        mentions = None
    return ExtractionResult(
        entry.custom_id.replace("_", "."),
        mentions,
        message.usage.input_tokens,
        message.usage.output_tokens,
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

    def _count(self, request: ExtractionRequest, model: str) -> int:
        return self._client.messages.count_tokens(**request_params(request, model)).input_tokens

    def count_tokens(self, requests: Sequence[ExtractionRequest], *, model: str) -> int:
        """Counts the batch in a few calls instead of one per paper.

        What every request shares (system prompt, schema, framing) is counted once, on a
        request with no text, and multiplied by the number of requests. The papers' own
        text is counted glued together in parts. Each part carries the shared part once
        more, so the sum is an overestimate, which is the safe side for a ceiling. The
        endpoint counts and does not bill.
        """
        overhead = self._count(ExtractionRequest("", "", ""), model)
        total = len(requests) * overhead
        part: list[str] = []
        size = 0
        for request in requests:
            text = _request_text(request)
            if part and size + len(text) > COUNT_CHARS_PER_CALL:
                total += self._count_part(part, model)
                part, size = [], 0
            part.append(text)
            size += len(text)
        if part:
            total += self._count_part(part, model)
        return total

    def _count_part(self, texts: list[str], model: str) -> int:
        # A part is one request whose message is the papers' texts one after another.
        return self._count(ExtractionRequest("", "", "\n\n".join(texts)), model)

    def submit_batch(self, requests: Sequence[ExtractionRequest], *, model: str) -> str:
        import anthropic

        entries = [
            {
                "custom_id": _custom_id(r.arxiv_id),
                "params": {**request_params(r, model), "max_tokens": EXTRACTION_MAX_OUTPUT_TOKENS},
            }
            for r in requests
        ]
        # No retries: the request carries no idempotency key, so a retry after a lost
        # answer would create a second batch, and pay for it.
        try:
            return (
                self._client.with_options(max_retries=0)
                .messages.batches.create(requests=entries)
                .id
            )
        except anthropic.APIStatusError as error:
            # These statuses came back before any batch was made. A timeout, a dropped
            # connection or a 5xx cannot be told from a batch that was made, so those are
            # not caught. A 429 is caught: without it a full run's estimate would stay
            # reserved as 'sending' for good.
            if error.status_code in _REFUSED_BEFORE_CREATING:
                raise BatchRejected(str(error)) from error
            raise

    def batch_ended(self, batch_id: str) -> bool:
        return self._client.messages.batches.retrieve(batch_id).processing_status == "ended"

    def batch_results(self, batch_id: str) -> Iterator[ExtractionResult]:
        for entry in self._client.messages.batches.results(batch_id):
            yield result_from_batch_entry(entry)
