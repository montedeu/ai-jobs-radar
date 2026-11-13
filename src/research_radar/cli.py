"""``radar``: the single entry point of the image; each Cloud Run resource runs one command."""

import argparse
import logging
from pathlib import Path

from research_radar.frequencies import DEFAULT_CATEGORY, DEFAULT_LIMIT

COMMANDS = {
    "migrate": "apply schema migrations up to the latest",
    "load-synonyms": "load the synonym dictionary file into the database, recording a version",
    "collect": "store the papers arXiv published or revised since the last run",
    "daily": "collect, then collect the sent batches and send the new papers to extraction",
    "backfill": "store the history of every slice, resuming where it stopped",
    "check-jobs": "fail unless a collection job could start (no request to arXiv)",
    "frequencies": "write the shares of the top entities of an extraction run to a Markdown file",
    "full-run": "estimate and, with --confirm, send the whole history to extraction (paid)",
    "abandon-batch": "give up a batch stuck in 'sending' or 'submitted' (its cost stays spent)",
    "measure": "compare an extraction run with the gold set and log the measurement",
    "measurements": "print the latest 'before' and 'after' measurements of extraction quality",
    "serve-mcp": "serve the MCP server on $PORT",
}


def _positive(text: str) -> int:
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _abandon_batch(connection, extraction, batch_id: int, confirm: bool) -> None:
    row = connection.execute(
        "SELECT run_id, status, estimated_cost_usd, provider_batch_id,"
        " (SELECT count(*) FROM extraction_batch_paper WHERE batch_id = b.id)"
        " FROM extraction_batch b WHERE id = %s",
        (batch_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"error: no extraction batch {batch_id}")
    run_id, status, estimate, provider_batch_id, papers = row
    print(f"Batch {batch_id} of run {run_id}: {status}, {papers} papers, estimated ${estimate}.")
    if status not in ("sending", "submitted"):
        raise SystemExit(f"error: only a sending or submitted batch can be abandoned, not {status}")
    print("The estimate stays in the spent total.")
    if status == "submitted":
        print(
            f"This batch did leave (provider batch id {provider_batch_id}). Giving it up means "
            "its results will not be collected and its papers will be sent again and paid for "
            "again. If it is still running at the provider, cancel it in the Anthropic console."
        )
    else:
        print(
            "Check the Anthropic console for a batch that was created: if this one did leave, "
            "its results will not be collected and its papers will be sent again and paid for "
            "twice."
        )
    if not confirm:
        print("Nothing was changed. Run again with --confirm to give it up.")
        return
    try:
        # Only the status that was just described: it may have changed since.
        extraction.abandon_batch(connection, batch_id, statuses=(status,))
    except (LookupError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"Abandoned batch {batch_id}: its {papers} papers are due again.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="radar")
    commands = parser.add_subparsers(dest="command", required=True)
    parsers = {name: commands.add_parser(name, help=summary) for name, summary in COMMANDS.items()}
    parsers["collect"].add_argument(
        "--max-papers",
        type=_positive,
        help="the limit on papers read in this run; raise it after the job was down for long",
    )
    parsers["daily"].add_argument(
        "--max-papers",
        type=_positive,
        help="the limit on papers read by the collection step; raise it after the job was "
        "down for long",
    )
    parsers["daily"].add_argument(
        "--max-extract-papers",
        type=_positive,
        help="the limit on papers sent to extraction in this run; raise it after a full run "
        "was switched to",
    )
    frequencies = parsers["frequencies"]
    frequencies.add_argument("--run", type=_positive, required=True, help="the extraction run id")
    frequencies.add_argument("--category", default=DEFAULT_CATEGORY, help="the slice")
    frequencies.add_argument("--limit", type=_positive, default=DEFAULT_LIMIT, help="rows")
    frequencies.add_argument("--out", type=Path, required=True, help="the Markdown file to write")
    parsers["full-run"].add_argument(
        "--confirm", action="store_true", help="send; without it only the estimate is shown"
    )
    parsers["full-run"].add_argument(
        "--restart",
        action="store_true",
        help="with --confirm: start a new full run even if an unfinished one exists",
    )
    parsers["full-run"].add_argument(
        "--abandon-sending",
        action="store_true",
        help="with --confirm: give up the batches left in 'sending' (their cost stays in the "
        "spent total) and continue",
    )
    abandon = parsers["abandon-batch"]
    abandon.add_argument("batch_id", type=_positive, help="the extraction batch id")
    abandon.add_argument(
        "--confirm", action="store_true", help="give the batch up; without it only describe it"
    )
    measure = parsers["measure"]
    measure.add_argument("--run", type=_positive, required=True, help="the extraction run id")
    measure.add_argument("--gold", type=Path, required=True, help="the gold set, JSON Lines")
    measure.add_argument(
        "--label",
        choices=("before", "after"),
        required=True,
        help="the gold set before or after the corrections of its labels",
    )
    args = parser.parse_args(argv)
    if args.command == "full-run":
        if (args.restart or args.abandon_sending) and not args.confirm:
            parser.error("--restart and --abandon-sending work only together with --confirm")
        if args.restart and args.abandon_sending:
            parser.error("--restart and --abandon-sending exclude each other")

    # Imported per command: a job does not pay for loading the web stack.
    if args.command == "migrate":
        from research_radar import migrate

        migrate.upgrade()
    elif args.command == "load-synonyms":
        from research_radar import database, synonyms

        # Connects like migrate and check-jobs, which run on the same job.
        with database.connect(database.direct_url(database.database_url())) as connection:
            try:
                count = synonyms.load_synonyms(connection)
            except ValueError as error:  # read_synonyms refused the file
                raise SystemExit(f"error: {error}") from error
        print(f"synonyms loaded: {count} entries")
    elif args.command == "collect":
        from research_radar import collect

        collect.run(max_papers=args.max_papers)
    elif args.command == "daily":
        from research_radar import collect, daily_extraction, extraction
        from research_radar.settings import EXTRACTION_DAILY_MAX_PAPERS

        # Each step has its own row in the run log and its own try: a failure of one must
        # not keep the other from running.
        failed = False
        steps = (
            lambda: collect.run(max_papers=args.max_papers),
            lambda: daily_extraction.run(
                client_factory=extraction.AnthropicModelClient,
                max_papers=args.max_extract_papers or EXTRACTION_DAILY_MAX_PAPERS,
            ),
        )
        for step in steps:
            try:
                step()
            except Exception:
                logging.exception("a step of the daily job failed")
                failed = True
        if failed:
            raise SystemExit(1)
    elif args.command == "check-jobs":
        from research_radar import collect

        collect.check()
    elif args.command == "backfill":
        from research_radar import collect

        collect.run_backfill()
    elif args.command == "frequencies":
        from research_radar import database, frequencies

        with database.connect() as connection:
            table = frequencies.top_entities(
                connection, args.run, category=args.category, limit=args.limit
            )
        frequencies.write_table(table, args.out)
    elif args.command == "full-run":
        from research_radar import database, extraction, full_run
        from research_radar.settings import EXTRACTION_MODEL

        # Counting tokens is free, so the estimate uses the real client even without
        # --confirm; sending and creating the run happen only with it, inside full_run.run.
        client = extraction.AnthropicModelClient()
        with database.connect(database.direct_url(database.database_url())) as connection:
            # The extraction steps refuse a connection with a transaction open.
            connection.autocommit = True
            try:
                plan = full_run.run(
                    connection,
                    client,
                    model=EXTRACTION_MODEL,
                    confirm=args.confirm,
                    restart=args.restart,
                    abandon_sending=args.abandon_sending,
                )
            except extraction.BudgetExceeded as error:
                raise SystemExit(f"error: {error}") from error
        if plan.refused:
            raise SystemExit(1)
    elif args.command == "abandon-batch":
        from research_radar import database, extraction

        with database.connect(database.direct_url(database.database_url())) as connection:
            # The extraction steps refuse a connection with a transaction open.
            connection.autocommit = True
            _abandon_batch(connection, extraction, args.batch_id, args.confirm)
    elif args.command == "measure":
        import hashlib

        from research_radar import database, quality

        try:
            # Read once: the hash in the log is of the bytes that were parsed.
            raw = args.gold.read_bytes()
            gold = quality.parse_gold(raw, args.gold.name)
        except (OSError, ValueError) as error:
            raise SystemExit(f"error: {error}") from error
        with database.connect() as connection:
            try:
                result = quality.measure(
                    connection,
                    args.run,
                    gold,
                    gold_sha256=hashlib.sha256(raw).hexdigest(),
                    label=args.label,
                )
            except ValueError as error:
                raise SystemExit(f"error: {error}") from error
        print(f"measurement logged ({result.label}, run {result.run_id})")
        print(quality.format_measurement(result))
    elif args.command == "measurements":
        from research_radar import database, quality

        with database.connect() as connection:
            print(quality.report(connection))
    elif args.command == "serve-mcp":
        from research_radar import mcp_server

        mcp_server.serve()
