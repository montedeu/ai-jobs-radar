"""``radar``: the single entry point of the image; each Cloud Run resource runs one command."""

import argparse

COMMANDS = {
    "migrate": "apply schema migrations up to the latest",
    "collect": "store the papers arXiv published or revised since the last run",
    "backfill": "store the history of every slice, resuming where it stopped",
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
    args = parser.parse_args(argv)

    # Imported per command: a job does not pay for loading the web stack.
    if args.command == "migrate":
        from research_radar import migrate

        migrate.upgrade()
    elif args.command == "collect":
        from research_radar import collect

        collect.run(max_papers=args.max_papers)
    elif args.command == "backfill":
        from research_radar import collect

        collect.run_backfill()
    elif args.command == "serve-mcp":
        from research_radar import mcp_server

        mcp_server.serve()
