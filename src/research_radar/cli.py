"""``radar``: the single entry point of the image; each Cloud Run resource runs one command."""

import argparse

COMMANDS = {
    "migrate": "apply schema migrations up to the latest",
    "collect": "run the daily collection job",
    "serve-mcp": "serve the MCP server on $PORT",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="radar")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, summary in COMMANDS.items():
        commands.add_parser(name, help=summary)
    args = parser.parse_args(argv)

    # Imported per command: a job does not pay for loading the web stack.
    if args.command == "migrate":
        from research_radar import migrate

        migrate.upgrade()
    elif args.command == "collect":
        from research_radar import collect

        collect.run()
    elif args.command == "serve-mcp":
        from research_radar import mcp_server

        mcp_server.serve()
