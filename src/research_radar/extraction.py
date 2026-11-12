"""Extraction: papers go to the model as one batch, the mentions come back into the database.

Submitting and collecting are separate steps, because a batch takes minutes to hours and
the process that sent it may be gone by then. Everything the second step needs is in the
database. The model is reached only through ModelClient; tests pass a fake one.
"""

import functools
import json
import logging
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Protocol

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from research_radar.mentions import PaperMentions
from research_radar.normalize import normal_form
from research_radar.papers import text_hash
from research_radar.settings import (
    BATCH_PRICES_USD_PER_MTOK,
    CATEGORIES,
    EXTRACTION_CEILING_USD,
    EXTRACTION_MAX_ATTEMPTS,
    EXTRACTION_MAX_OUTPUT_TOKENS,
    HISTORY_START,
    REASONING_BY_DEFAULT_MODELS,
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


# The user message of a request; {title} and {abstract} stand for the paper's own text.
MESSAGE_TEMPLATE = "Title: {title}\n\nAbstract: {abstract}"


@dataclass(frozen=True)
class RequestConfig:
    """What a run fixes about its requests besides the model and the texts."""

    # False: the request switches the model's reasoning off.
    reasoning: bool = True
    max_output_tokens: int = EXTRACTION_MAX_OUTPUT_TOKENS


DEFAULT_CONFIG = RequestConfig()


class RunConfigMismatch(Exception):
    """The code would send requests that differ from the ones the run was started with."""


class _StoredConfig(BaseModel):
    """extraction_run.request_config as read back: every field present, of exactly its type."""

    model_config = ConfigDict(extra="forbid", strict=True)

    reasoning: bool
    max_output_tokens: int = Field(gt=0)


@functools.cache
def _output_schema() -> dict[str, Any]:
    # The model's answer is held to the schema; transform_schema is the SDK's way to make
    # a pydantic schema one the API accepts.
    from anthropic import transform_schema

    return transform_schema(PaperMentions)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: RequestConfig, model: str) -> str:
    """The hash of the request a run sends, apart from the paper's text and the model.

    It is the hash of the very parameters that go into a batch entry, built by
    request_params from a stand-in paper whose title and abstract are the placeholders, so
    nothing that is sent can change without the hash changing. The model is left out: the
    run records it itself.
    """
    stand_in = ExtractionRequest("", "{title}", "{abstract}")
    params = request_params(stand_in, model, config)
    del params["model"]
    params["max_tokens"] = config.max_output_tokens
    return text_hash(_canonical_json(params))


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
    # How a request that got no answer ended: "errored", "expired" or "canceled". None for
    # a request the model answered, usable or not.
    unanswered: str | None = None


class BatchRejected(Exception):
    """The API refused to create the batch outright: nothing was created, nothing is billed.

    Any other failure of submit_batch (a timeout, a dropped connection, a 5xx) leaves it
    unknown whether the batch exists, and must propagate as it is.
    """


class ModelClient(Protocol):
    """The only place the model is called."""

    def count_tokens(
        self, requests: Sequence[ExtractionRequest], *, model: str, config: RequestConfig
    ) -> int: ...

    def submit_batch(
        self, requests: Sequence[ExtractionRequest], *, model: str, config: RequestConfig
    ) -> str:
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


# The papers a full run is meant to cover: the slices since HISTORY_START, with an id that
# can be extracted at all. start_run snapshots it; the full-run plan counts it before any
# run exists.
SCOPE_SQL = """
    SELECT p.arxiv_id FROM paper p
     WHERE p.first_version_date >= %(start)s
       AND p.arxiv_id ~ %(shape)s
       AND EXISTS (SELECT 1 FROM paper_category c
                    WHERE c.arxiv_id = p.arxiv_id AND c.category = ANY(%(slices)s))
"""
SCOPE_PARAMS = {
    "start": HISTORY_START,
    "shape": f"^{_ARXIV_ID.pattern}$",
    "slices": list(CATEGORIES),
}


def start_run(
    connection: psycopg.Connection,
    *,
    model: str,
    config: RequestConfig = DEFAULT_CONFIG,
    note: str = "",
    full: bool = False,
) -> int:
    """Creates a run pinned to the code's request configuration (see config_hash).

    A full run is one meant to cover the whole history (see runs.py).

    A full run also records its scope in the same transaction: the papers of the slices
    since HISTORY_START that exist now, with an id that can be extracted at all. Papers
    that come later, or gain a slice later, are not in it; the daily job sends them to
    the current run.
    """
    require_idle(connection)
    if config.reasoning and model not in REASONING_BY_DEFAULT_MODELS:
        # With reasoning on, the request sends no thinking parameter and relies on the
        # model's default; that is known only for the models in the set.
        raise ValueError(
            f"model {model!r} is not known to reason by default; add it to "
            "REASONING_BY_DEFAULT_MODELS in settings.py, or start the run without reasoning"
        )
    with connection.transaction():
        run_id = connection.execute(
            "INSERT INTO extraction_run (model, prompt_hash, request_config, note, is_full)"
            " VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (model, config_hash(config, model), Jsonb(asdict(config)), note, full),
        ).fetchone()[0]
        if full:
            connection.execute(
                "INSERT INTO extraction_run_scope (run_id, arxiv_id)"
                " SELECT %(run)s, s.arxiv_id FROM (" + SCOPE_SQL + ") s",
                {**SCOPE_PARAMS, "run": run_id},
            )
    return run_id


def estimate_usd(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    price_in, price_out = BATCH_PRICES_USD_PER_MTOK[model]
    return (Decimal(input_tokens) * price_in + Decimal(output_tokens) * price_out) / 1_000_000


# The papers a run has given up on, with the hash of the abstract the attempts were made on:
# those with EXTRACTION_MAX_ATTEMPTS paid, unusable answers ('invalid') or API rejections
# ('errored') to the abstract they have now. Expired and canceled requests do not count.
# One grouped pass over the run's batches. _due and runs.py both use it, so a paper the
# full run stops waiting for is also one the daily job stops paying for.
GIVEN_UP_SQL = """
    SELECT bp.arxiv_id, bp.abstract_hash
      FROM extraction_batch b
      JOIN extraction_batch_paper bp ON bp.batch_id = b.id
      JOIN paper p ON p.arxiv_id = bp.arxiv_id AND p.abstract_hash = bp.abstract_hash
     WHERE b.run_id = %(run)s AND b.status = 'collected' AND bp.outcome IN ('invalid', 'errored')
     GROUP BY bp.arxiv_id, bp.abstract_hash
    HAVING count(*) >= %(attempts)s
"""


# The papers a run has out in a batch that is not collected: still sending (nobody knows
# whether it left) or submitted. _due leaves them out; the full-run plan counts them.
IN_FLIGHT_SQL = """
    SELECT bp.arxiv_id, bp.abstract_hash, b.id AS batch_id, b.status
      FROM extraction_batch_paper bp JOIN extraction_batch b ON b.id = bp.batch_id
     WHERE b.run_id = %(run)s AND b.status IN ('sending', 'submitted')
"""


def require_idle(connection: psycopg.Connection) -> None:
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


def abandon_batch(connection: psycopg.Connection, batch_id: int) -> int:
    """Gives up a batch that is 'sending' or 'submitted'; returns the number of its papers.

    The batch stays in spent_usd at its estimate: nobody knows whether it left, or what a
    submitted one will still cost. Its papers stop counting as in flight, so they are due
    again; if the batch did go out, they are paid for twice. collect leaves it alone.
    Raises LookupError for an unknown batch and ValueError for one in another status.
    The connection must have no transaction open.
    """
    require_idle(connection)
    with connection.transaction():
        # The update takes the row lock, so a collector storing the batch at this moment
        # finishes first (and then this finds it 'collected') or sees 'abandoned' after.
        given_up = connection.execute(
            "UPDATE extraction_batch SET status = 'abandoned'"
            " WHERE id = %s AND status IN ('sending', 'submitted') RETURNING id",
            (batch_id,),
        ).fetchone()
        if given_up is None:
            row = connection.execute(
                "SELECT status FROM extraction_batch WHERE id = %s", (batch_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"no extraction batch {batch_id}")
            raise ValueError(
                f"batch {batch_id} is {row[0]}; only a sending or submitted batch can be abandoned"
            )
        return connection.execute(
            "SELECT count(*) FROM extraction_batch_paper WHERE batch_id = %s", (batch_id,)
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
    Raises RunConfigMismatch, before anything is counted or recorded, when the code's request
    configuration is no longer the one the run was started with. Raises BudgetExceeded, before
    anything is sent, when the batch's most expensive outcome
    plus what is spent passes the ceiling. Ids of a shape that cannot be sent (see
    _ARXIV_ID) are left out of the batch and logged as a warning.

    The batch is recorded as 'sending' before it is sent: a failure after that leaves the
    row, counted in the budget, instead of a paid batch nobody knows about. Only a refusal
    that certainly created nothing (BatchRejected) takes the money back and frees the papers.
    """
    require_idle(connection)
    arxiv_ids = _supported(arxiv_ids)
    if not arxiv_ids:
        return None
    with connection.transaction():
        model, config = _run_settings(connection, run_id)
        todo = _due(connection, run_id, arxiv_ids)
    if not todo:
        return None
    for _attempt in range(MAX_COUNTS):
        # Counting is slow network work and takes no money; it stays outside the lock below.
        input_tokens = client.count_tokens(
            [request for request, _ in todo], model=model, config=config
        )
        # The output is counted at the cap for every paper: the real size is not known before.
        estimate = estimate_usd(model, input_tokens, len(todo) * config.max_output_tokens)
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
        provider_batch_id = client.submit_batch(
            [request for request, _ in todo], model=model, config=config
        )
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


def _run_settings(connection: psycopg.Connection, run_id: int) -> tuple[str, RequestConfig]:
    """The run's model and request configuration, once the code is known to match it."""
    row = connection.execute(
        "SELECT model, prompt_hash, request_config FROM extraction_run WHERE id = %s", (run_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"no extraction run {run_id}")
    model, pinned_hash, stored = row
    if model not in BATCH_PRICES_USD_PER_MTOK:
        raise ValueError(f"no batch prices for model {model!r}; add them to settings.py")
    try:
        config = RequestConfig(**_StoredConfig.model_validate(stored).model_dump())
    except ValidationError as error:
        # An empty object is what a run from before migration 0007 has; there is no such
        # run to be kept alive, so it is an error like any other bad value.
        raise RunConfigMismatch(
            f"run {run_id} has no valid request configuration ({stored!r}); start a new run"
        ) from error
    if config_hash(config, model) != pinned_hash:
        raise RunConfigMismatch(
            f"run {run_id} was started with another prompt, template, schema or request "
            "configuration than the code has now; start a new run"
        )
    return model, config


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
            SELECT f.arxiv_id, f.abstract_hash FROM ("""
            + IN_FLIGHT_SQL
            + """) f
             WHERE f.arxiv_id = ANY(%(ids)s)
            UNION
            SELECT g.arxiv_id, g.abstract_hash FROM ("""
            + GIVEN_UP_SQL
            + """) g
             WHERE g.arxiv_id = ANY(%(ids)s)
            """,
            {"run": run_id, "ids": ids, "attempts": EXTRACTION_MAX_ATTEMPTS},
        ).fetchall()
    )
    return [stored[i] for i in ids if (i, stored[i][1]) not in taken]


def collect(connection: psycopg.Connection, client: ModelClient, batch_id: int) -> Counter:
    """Stores a finished batch's answers.

    Counts "extracted", "mentions", "pending" (not ended yet), "unsent" (stuck in
    'sending', for a person to sort out), "abandoned" (a stuck batch a person gave up on:
    never fetched), "rejected" (the API refused the batch), "failed"
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
    require_idle(connection)
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
    if status in ("sending", "rejected", "abandoned"):
        return Counter({{"sending": "unsent"}.get(status, status): papers})
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
    outcomes: list[tuple[str, int, str]] = []
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
            # Only a paid answer not in the schema ("invalid") and a request the API
            # rejects ("errored") count toward giving up; expired and canceled ones say
            # nothing about the paper.
            outcome = result.unanswered or "invalid"
        elif current.get(result.arxiv_id) != sent[result.arxiv_id]:
            # The abstract changed after the send; a batch sent for the newer one may be
            # stored already, and this answer must not replace it.
            counter["stale"] += 1
            outcome = "stale"
        else:
            counter["extracted"] += 1
            counter["mentions"] += _store(connection, run_id, sent[result.arxiv_id], result)
            outcome = "extracted"
        outcomes.append((outcome, batch_id, result.arxiv_id))
    with connection.cursor() as cursor:
        cursor.executemany(
            "UPDATE extraction_batch_paper SET outcome = %s WHERE batch_id = %s AND arxiv_id = %s",
            outcomes,
        )
    connection.execute(
        """
        UPDATE extraction_batch
           SET status = 'collected', collected_at = now(), actual_cost_usd = %s
         WHERE id = %s
        """,
        (estimate_usd(model, input_tokens, output_tokens), batch_id),
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


def request_params(request: ExtractionRequest, model: str, config: RequestConfig) -> dict[str, Any]:
    """The Messages API parameters for one paper; a batch entry and a token count share them."""
    params: dict[str, Any] = {
        "model": model,
        "system": EXTRACTION_PROMPT,
        "messages": [{"role": "user", "content": _request_text(request)}],
        # Low effort: reading an abstract for names needs little thinking, and thinking
        # tokens are billed as output.
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": _output_schema()},
        },
    }
    if not config.reasoning:
        # Claude Haiku 5.5 thinks unless told not to; "disabled" is accepted at effort high
        # or below, and this request is at low. Haiku 5.5 has no budget_tokens.
        params["thinking"] = {"type": "disabled"}
    return params


def _request_text(request: ExtractionRequest) -> str:
    return MESSAGE_TEMPLATE.format(title=request.title, abstract=request.abstract)


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
        return ExtractionResult(
            entry.custom_id.replace("_", "."), None, 0, 0, unanswered=entry.result.type
        )
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

        # The key comes from an environment variable. Putting it in Secret Manager for Cloud Run
        # is not done yet; see #22.
        self._client = anthropic.Anthropic()

    def _count(self, request: ExtractionRequest, model: str, config: RequestConfig) -> int:
        params = request_params(request, model, config)
        return self._client.messages.count_tokens(**params).input_tokens

    def count_tokens(
        self, requests: Sequence[ExtractionRequest], *, model: str, config: RequestConfig
    ) -> int:
        """Counts the batch in a few calls instead of one per paper.

        What every request shares (system prompt, schema, framing) is counted once, on a
        request with no text, and multiplied by the number of requests. The papers' own
        text is counted glued together in parts. Each part carries the shared part once
        more, so the sum is an overestimate, which is the safe side for a ceiling. The
        endpoint counts and does not bill.
        """
        overhead = self._count(ExtractionRequest("", "", ""), model, config)
        total = len(requests) * overhead
        part: list[str] = []
        size = 0
        for request in requests:
            text = _request_text(request)
            if part and size + len(text) > COUNT_CHARS_PER_CALL:
                total += self._count_part(part, model, config)
                part, size = [], 0
            part.append(text)
            size += len(text)
        if part:
            total += self._count_part(part, model, config)
        return total

    def _count_part(self, texts: list[str], model: str, config: RequestConfig) -> int:
        # A part is one request whose message is the papers' texts one after another.
        return self._count(ExtractionRequest("", "", "\n\n".join(texts)), model, config)

    def submit_batch(
        self, requests: Sequence[ExtractionRequest], *, model: str, config: RequestConfig
    ) -> str:
        import anthropic

        entries = [
            {
                "custom_id": _custom_id(r.arxiv_id),
                "params": {
                    **request_params(r, model, config),
                    "max_tokens": config.max_output_tokens,
                },
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
