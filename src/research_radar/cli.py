"""``radar``: the single entry point of the image; each Cloud Run resource runs one command."""

import argparse
from pathlib import Path

COMMANDS = {
    "migrate": "apply schema migrations up to the latest",
    "collect": "store the papers arXiv published or revised since the last run",
    "backfill": "store the history of every slice, resuming where it stopped",
    "check-jobs": "fail unless a collection job could start (no request to arXiv)",
    "frequencies": "write the most frequent entities of an extraction run to a Markdown file",
    "serve-mcp": "serve the MCP server on $PORT",
}


def _positive(text: str) -> int:
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="radar")
    commands = parser.add_subparsers(dest="command", required=True)
    parsers = {name: commands.add_parser(name, help=summary) for name, summary in COMMANDS.items()}
    parsers["collect"].add_argument(
        "--max-papers",
        type=_positive,
        help="the limit on papers read in this run; raise it after the job was down for long",
    )
    frequencies = parsers["frequencies"]
    frequencies.add_argument("--run", type=_positive, required=True, help="the extraction run id")
    frequencies.add_argument("--category", default="cs.CL", help="the slice (default cs.CL)")
    frequencies.add_argument("--limit", type=_positive, default=40, help="rows (default 40)")
    frequencies.add_argument("--out", type=Path, required=True, help="the Markdown file to write")
    args = parser.parse_args(argv)

    # Imported per command: a job does not pay for loading the web stack.
    if args.command == "migrate":
        from research_radar import migrate

        migrate.upgrade()
    elif args.command == "collect":
        from research_radar import collect

        collect.run(max_papers=args.max_papers)
    elif args.command == "check-jobs":
        from research_radar import collect

        collect.check()
    elif args.command == "backfill":
        from research_radar import collect

        collect.run_backfill()
    elif args.command == "frequencies":
        from research_radar import database, frequencies

        with database.connect() as connection:
            rows = frequencies.top_entities(
                connection, args.run, category=args.category, limit=args.limit
            )
            completed = frequencies.completed_papers(connection, args.run, args.category)
        frequencies.write_table(
            rows, args.out, run_id=args.run, category=args.category, completed=completed
        )
    elif args.command == "serve-mcp":
        from research_radar import mcp_server

        mcp_server.serve()
