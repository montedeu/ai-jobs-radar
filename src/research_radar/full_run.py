"""The full run: the whole collected history goes to extraction as one new run, in parts.

Only the developer starts it (CLAUDE.md, "Деньги"). Without confirmation it prints the plan
and the cost estimate and sends nothing. A part is one extraction batch, so a stop leaves
the parts already sent in the database and the next start continues the same run. Collecting
the answers and making the run current is the daily job's work (runs.promote).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

import psycopg

from research_radar import extraction, runs
from research_radar.extraction import (
    DEFAULT_CONFIG,
    BudgetExceeded,
    ExtractionRequest,
    ModelClient,
    RequestConfig,
)
from research_radar.settings import (
    BATCH_PRICES_USD_PER_MTOK,
    EXTRACTION_BATCH_SIZE,
    EXTRACTION_CEILING_USD,
    EXTRACTION_MAX_ATTEMPTS,
    EXTRACTION_MAX_GIVEN_UP_SHARE,
)

READY = "ready"
EMPTY_HISTORY = "empty_history"
ALREADY_CURRENT = "already_current"
NOTHING_TO_SEND = "nothing_to_send"
BLOCKED = "blocked"


@dataclass(frozen=True)
class FullRunPlan:
    run_id: int | None  # the run that will be continued; None: a new one will be created
    papers: int  # to send: the scope minus the marked, the given-up and those in a batch
    batches: int
    estimated_usd: Decimal  # at batch prices, the output counted at the cap
    spent_usd: Decimal
    ceiling_usd: Decimal
    in_flight: int = 0  # in a batch that is sending or sent and not collected
    given_up: int = 0  # the run gave up on them (extraction.GIVEN_UP_SQL)
    scope: int = 0
    status: str = READY


# The run to continue is one that runs.promote could still make current, with the request
# this start would send: any other would be paid for and then never switched to.
_CONTINUABLE_SQL = """
    SELECT id FROM extraction_run
     WHERE is_full AND current_since IS NULL AND model = %s AND prompt_hash = %s
       AND created_at > coalesce(
             (SELECT created_at FROM extraction_run WHERE id = %s), '-infinity')
     ORDER BY created_at DESC, id DESC LIMIT 1
"""

_IN_FLIGHT_SQL = """
    SELECT DISTINCT bp.arxiv_id
      FROM extraction_batch b JOIN extraction_batch_paper bp ON bp.batch_id = b.id
     WHERE b.run_id = %s AND b.status IN ('sending', 'submitted')
"""

_GIVEN_UP_COUNT_SQL = (
    """
    SELECT count(*) FROM (SELECT DISTINCT g.arxiv_id FROM ("""
    + extraction.GIVEN_UP_SQL
    + """) g
     JOIN extraction_run_scope s ON s.run_id = %(run)s AND s.arxiv_id = g.arxiv_id
     WHERE NOT EXISTS (SELECT 1 FROM paper_extraction e
                        WHERE e.run_id = s.run_id AND e.arxiv_id = s.arxiv_id)) given
"""
)


def plan(
    connection: psycopg.Connection,
    client: ModelClient,
    *,
    model: str,
    config: RequestConfig = DEFAULT_CONFIG,
) -> FullRunPlan:
    """What a full run would do now; reads and counts tokens, writes nothing."""
    return _prepare(connection, client, model, config)[0]


def _prepare(
    connection: psycopg.Connection, client: ModelClient, model: str, config: RequestConfig
) -> tuple[FullRunPlan, list[str]]:
    if model not in BATCH_PRICES_USD_PER_MTOK:
        # Checked here and not at the first submit: by then start_run would have left a full
        # run with a snapshot that nothing can pay for.
        raise ValueError(f"no batch prices for model {model!r}; add them to settings.py")
    spent = extraction.spent_usd(connection)

    def result(status: str, ids: list[str] = (), **fields) -> tuple[FullRunPlan, list[str]]:
        estimated = Decimal(0)
        if ids:
            requests = _requests(connection, ids)
            input_tokens = client.count_tokens(requests, model=model, config=config)
            # The output is counted at the cap for every paper, as submit does.
            estimated = extraction._usd(model, input_tokens, len(ids) * config.max_output_tokens)
        batches = math.ceil(len(ids) / EXTRACTION_BATCH_SIZE)
        found = FullRunPlan(
            fields.pop("run_id", None),
            len(ids),
            batches,
            estimated,
            spent,
            EXTRACTION_CEILING_USD,
            status=status,
            **fields,
        )
        return found, list(ids)

    pinned = extraction.config_hash(config, model)
    current = runs.current_run(connection)
    if current is not None:
        same = connection.execute(
            "SELECT model = %s AND prompt_hash = %s FROM extraction_run WHERE id = %s",
            (model, pinned, current),
        ).fetchone()[0]
        if same:
            # Another start would pay for the whole history again.
            return result(ALREADY_CURRENT)
    row = connection.execute(_CONTINUABLE_SQL, (model, pinned, current)).fetchone()
    if row is None:
        scope = sorted(
            r[0] for r in connection.execute(extraction.SCOPE_SQL, extraction.SCOPE_PARAMS)
        )
        if not scope:
            return result(EMPTY_HISTORY)
        return result(READY, scope, scope=len(scope))
    run_id = row[0]
    missing = runs.missing_papers(connection, run_id)
    in_flight = {r[0] for r in connection.execute(_IN_FLIGHT_SQL, (run_id,))}
    given_up = connection.execute(
        _GIVEN_UP_COUNT_SQL, {"run": run_id, "attempts": EXTRACTION_MAX_ATTEMPTS}
    ).fetchone()[0]
    scope_size = connection.execute(
        "SELECT count(*) FROM extraction_run_scope WHERE run_id = %s", (run_id,)
    ).fetchone()[0]
    to_send = [i for i in missing if i not in in_flight]
    fields = {
        "run_id": run_id,
        "in_flight": len(in_flight),
        "given_up": given_up,
        "scope": scope_size,
    }
    if to_send:
        return result(READY, to_send, **fields)
    if not missing and given_up > max(1, EXTRACTION_MAX_GIVEN_UP_SHARE * scope_size):
        # The same limit as runs.promote: the run covered its scope and still will not switch.
        return result(BLOCKED, **fields)
    return result(NOTHING_TO_SEND, **fields)


def _requests(connection: psycopg.Connection, ids: list[str]) -> list[ExtractionRequest]:
    rows = {
        row[0]: ExtractionRequest(*row)
        for row in connection.execute(
            "SELECT arxiv_id, title, abstract FROM paper WHERE arxiv_id = ANY(%s)", (ids,)
        )
    }
    return [rows[i] for i in ids]


def run(
    connection: psycopg.Connection,
    client: ModelClient,
    *,
    model: str,
    config: RequestConfig = DEFAULT_CONFIG,
    confirm: bool,
    out: Callable[[str], None] = print,
) -> FullRunPlan:
    """Prints the plan; with confirm, creates or continues the run and sends it in parts.

    The ceiling is submit's own check, part by part. A stop at the ceiling after some
    parts is normal for a history this size: the rest waits until the daily job has
    collected the parts sent. A stop before the first part raises BudgetExceeded.
    """
    extraction.require_idle(connection)
    found, ids = _prepare(connection, client, model, config)
    _print_plan(found, out)
    if found.status != READY:
        return found
    if not confirm:
        out(
            "Nothing was sent. Sending the full history spends money (ceiling "
            f"${found.ceiling_usd}); run again with --confirm to send it."
        )
        return found
    if found.run_id is None:
        run_id = extraction.start_run(connection, model=model, config=config, full=True)
        out(f"Created full run {run_id}.")
        # The snapshot is taken now, so papers collected since the plan are in it.
        ids = runs.missing_papers(connection, run_id)
    else:
        run_id = found.run_id
        out(f"Continuing full run {run_id}.")
    parts = [ids[i : i + EXTRACTION_BATCH_SIZE] for i in range(0, len(ids), EXTRACTION_BATCH_SIZE)]
    sent = 0
    for part in parts:
        try:
            if extraction.submit(connection, client, run_id, part) is not None:
                sent += 1
        except BudgetExceeded as error:
            if sent == 0:
                raise
            out(
                f"Stopped at the ceiling: {sent} of {len(parts)} parts sent, "
                f"{len(parts) - sent} left. Run the command again after the daily job has "
                f"collected the sent batches. ({error})"
            )
            return found
        except Exception:
            out(f"Failed: {sent} of {len(parts)} parts sent. Run the command again to continue.")
            raise
    out(f"Sent {sent} of {len(parts)} parts of run {run_id}.")
    return found


def _print_plan(found: FullRunPlan, out: Callable[[str], None]) -> None:
    if found.status == EMPTY_HISTORY:
        out("There is no collected history to extract: no paper is in scope, nothing to send.")
        return
    if found.status == ALREADY_CURRENT:
        out(
            "A full run with this model and request configuration is already current: "
            "nothing to send."
        )
        return
    if found.status == BLOCKED:
        out(
            f"The run {found.run_id} covered its scope but gave up on {found.given_up} of "
            f"{found.scope} papers, so it will not become current. Nothing to send; "
            "look at the failed papers."
        )
        return
    target = "a new run" if found.run_id is None else f"run {found.run_id}"
    out(
        f"Full run: {found.papers} papers to send in {found.batches} batches ({target}); "
        f"in flight: {found.in_flight}; given up: {found.given_up}."
    )
    if found.status == NOTHING_TO_SEND:
        out("There is nothing to send: every paper of the run is marked, in flight or given up.")
        return
    out(
        f"Cost estimated at ${found.estimated_usd:.4f}; spent so far ${found.spent_usd:.4f}; "
        f"ceiling ${found.ceiling_usd}."
    )
