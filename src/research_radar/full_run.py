"""The full run: the whole collected history goes to extraction as one new run, in parts.

Only the developer starts it (CLAUDE.md, "Деньги"). Without confirmation it prints the plan
and the cost estimate and sends nothing. A part is one extraction batch, so a stop leaves
the parts already sent in the database and the next start continues the same run. Collecting
the answers and making the run current is the daily job's work (runs.promote).
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from decimal import Decimal

import psycopg

from research_radar import database, extraction, runs
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
)

READY = "ready"
EMPTY_HISTORY = "empty_history"
ALREADY_CURRENT = "already_current"
NOTHING_TO_SEND = "nothing_to_send"
BLOCKED = "blocked"
NEWER_RUN_OTHER_CONFIG = "newer_run_other_config"
LOCKED = "locked"
STUCK_SENDING = "stuck_sending"

log = logging.getLogger(__name__)

# One command at a time: two would each count what is in flight before either sent.
LOCK_KEY = "full_run"


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
    sending: int = 0  # batches stuck in 'sending': nobody knows whether they left
    status: str = READY
    other_run: int | None = None  # the run that stands in the way (NEWER_RUN_OTHER_CONFIG)

    @property
    def refused(self) -> bool:
        """The command can do nothing useful, and the caller should fail."""
        return self.status in (
            EMPTY_HISTORY,
            BLOCKED,
            NEWER_RUN_OTHER_CONFIG,
            LOCKED,
            STUCK_SENDING,
        )


_RESTART_HINT = (
    "Sort out the batches stuck in 'sending', or run again with --confirm --restart: "
    "that starts a new full run, which is newer, so the old one never becomes current."
)


def plan(
    connection: psycopg.Connection,
    client: ModelClient,
    *,
    model: str,
    config: RequestConfig = DEFAULT_CONFIG,
    restart: bool = False,
) -> FullRunPlan:
    """What a full run would do now; reads and counts tokens, writes nothing."""
    return _prepare(connection, client, model, config, restart)[0]


def _prepare(
    connection: psycopg.Connection,
    client: ModelClient,
    model: str,
    config: RequestConfig,
    restart: bool,
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
            estimated = extraction.estimate_usd(
                model, input_tokens, len(ids) * config.max_output_tokens
            )
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
    if current is not None and not restart:
        same = connection.execute(
            "SELECT model = %s AND prompt_hash = %s FROM extraction_run WHERE id = %s",
            (model, pinned, current),
        ).fetchone()[0]
        if same:
            # Another start would pay for the whole history again.
            return result(ALREADY_CURRENT)
    head = None if restart else runs.head_candidate(connection)
    if head is None:
        scope = sorted(
            r[0] for r in connection.execute(extraction.SCOPE_SQL, extraction.SCOPE_PARAMS)
        )
        if not scope:
            return result(EMPTY_HISTORY)
        return result(READY, scope, scope=len(scope))
    run_model, run_hash = connection.execute(
        "SELECT model, prompt_hash FROM extraction_run WHERE id = %s", (head,)
    ).fetchone()
    if (run_model, run_hash) != (model, pinned):
        # Paying for an older run is pointless: the head is newer, and promote never
        # switches to a run older than the current one.
        return result(NEWER_RUN_OTHER_CONFIG, other_run=head)
    state = runs.run_state(connection, head)
    in_flight, sending = _in_flight(connection, head)
    fields = {
        "run_id": head,
        "in_flight": len(in_flight),
        "given_up": state.given_up,
        "scope": state.scope,
        "sending": sending,
    }
    if sending:
        # A batch in 'sending' may or may not have left; more parts on top of it could pay
        # twice for what it carries.
        return result(STUCK_SENDING, **fields)
    if state.blocked:
        # Whatever is still missing, the run will not become current: sending more is wasted.
        return result(BLOCKED, **fields)
    to_send = [i for i in runs.missing_papers(connection, head) if i not in in_flight]
    if to_send:
        return result(READY, to_send, **fields)
    return result(NOTHING_TO_SEND, **fields)


def _in_flight(connection: psycopg.Connection, run_id: int) -> tuple[set[str], int]:
    """The papers the run has in an uncollected batch, and how many of its batches are
    stuck in 'sending'."""
    rows = connection.execute(extraction.IN_FLIGHT_SQL, {"run": run_id}).fetchall()
    return {row[0] for row in rows}, len({row[2] for row in rows if row[3] == "sending"})


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
    restart: bool = False,
    out: Callable[[str], None] = print,
) -> FullRunPlan:
    """Prints the plan; with confirm, creates or continues the run and sends it in parts.

    The ceiling is submit's own check, part by part. A stop at the ceiling after some
    parts is normal for a history this size: the rest waits until the daily job has
    collected the parts sent. A stop before the first part raises BudgetExceeded.
    """
    database.require_direct(connection)
    extraction.require_idle(connection)
    if restart and not confirm:
        raise ValueError("restart needs confirm")
    if not connection.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (LOCK_KEY,)).fetchone()[
        0
    ]:
        out("Another full-run command is already running; nothing was done.")
        return FullRunPlan(
            None, 0, 0, Decimal(0), Decimal(0), EXTRACTION_CEILING_USD, status=LOCKED
        )
    try:
        return _run_locked(connection, client, model, config, confirm, restart, out)
    finally:
        # A broken connection took the lock with it; failing here would hide the error
        # that broke it.
        if not connection.closed:
            try:
                connection.execute("SELECT pg_advisory_unlock(hashtext(%s))", (LOCK_KEY,))
            except psycopg.Error:
                log.warning("could not release the full-run lock", exc_info=True)


def _run_locked(
    connection: psycopg.Connection,
    client: ModelClient,
    model: str,
    config: RequestConfig,
    confirm: bool,
    restart: bool,
    out: Callable[[str], None],
) -> FullRunPlan:
    found, ids = _prepare(connection, client, model, config, restart)
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
    out(f"Sending {len(ids)} papers in {len(parts)} parts.")
    sent = skipped = 0
    for index, part in enumerate(parts):
        stop = _stop_before_part(connection, run_id)
        if stop is not None:
            _print_stop(found, stop, run_id, sent, skipped, len(parts) - index, out)
            return replace(found, status=stop, run_id=run_id)
        try:
            if extraction.submit(connection, client, run_id, part) is None:
                skipped += 1
            else:
                sent += 1
        except BudgetExceeded as error:
            if sent == 0 and skipped == 0:
                raise
            out(
                f"Stopped at the ceiling: sent {sent}, skipped {skipped}, "
                f"left {len(parts) - index}. Run the command again after the daily job has "
                f"collected the sent batches. ({error})"
            )
            return found
        except Exception:
            out(
                f"Failed: sent {sent}, skipped {skipped}, left {len(parts) - index}. "
                "Run the command again to continue."
            )
            raise
    out(
        f"Done: sent {sent}, skipped {skipped}, left 0. A skipped part had everything "
        "already in flight or done."
    )
    return found


def _stop_before_part(connection: psycopg.Connection, run_id: int) -> str | None:
    """Why no further part may be sent: what was true at the plan can change under a long
    send (a collected batch can push the run over the limit, a send can be left hanging)."""
    if _in_flight(connection, run_id)[1]:
        return STUCK_SENDING
    if runs.run_state(connection, run_id).blocked:
        return BLOCKED
    return None


def _print_stop(
    found: FullRunPlan,
    stop: str,
    run_id: int,
    sent: int,
    skipped: int,
    left: int,
    out: Callable[[str], None],
) -> None:
    why = (
        "a batch of the run is stuck in 'sending'"
        if stop == STUCK_SENDING
        else "the run gave up on too many papers and will not become current"
    )
    out(f"Stopped before the next part of run {run_id}: {why}. {_RESTART_HINT}")
    out(f"Sent {sent}, skipped {skipped}, left {left}.")


def _print_plan(found: FullRunPlan, out: Callable[[str], None]) -> None:
    if found.status == EMPTY_HISTORY:
        out("There is no collected history to extract: no paper is in scope, nothing to send.")
        return
    if found.status == ALREADY_CURRENT:
        out(
            "A full run with this model and request configuration is already current: "
            "nothing to send. To pay for the whole history again anyway (for example after "
            "adding a slice to CATEGORIES, which the configuration hash does not cover), "
            "run with --confirm --restart."
        )
        return
    if found.status == NEWER_RUN_OTHER_CONFIG:
        out(
            f"There is a newer unfinished full run {found.other_run} with another "
            "configuration; continuing the older one is pointless, and nothing was created."
            f" {_RESTART_HINT}"
        )
        return
    if found.status == STUCK_SENDING:
        out(
            f"The run {found.run_id} has {found.sending} batches stuck in 'sending': they may "
            f"or may not have left, so nothing more is sent. {_RESTART_HINT}"
        )
        return
    if found.status == BLOCKED:
        out(
            f"The run {found.run_id} gave up on {found.given_up} of {found.scope} papers, "
            f"so it will not become current; nothing is sent. {_RESTART_HINT}"
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
        f"ceiling ${found.ceiling_usd}. The reservation part by part is slightly higher "
        "(each part carries its own overhead) and is counted again when a part is sent."
    )
