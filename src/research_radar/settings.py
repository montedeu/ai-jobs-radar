"""Fixed choices of the project that code in several modules needs."""

from datetime import date
from decimal import Decimal

# Slices that are collected. Adding one here is all the collector needs; the cost of
# extracting the new papers is a separate decision (CLAUDE.md).
CATEGORIES = ("cs.CL",)

# The history starts here and only grows; nothing older is stored (docs/adr/0003).
HISTORY_START = date(2025, 11, 1)

# Extraction (docs/adr/0005; the ceiling is CLAUDE.md's $20 for the whole extraction).
EXTRACTION_MODEL = "claude-haiku-5-5"
EXTRACTION_CEILING_USD = Decimal(20)

# Message Batches price per model, USD per million tokens (input, output). Claude Haiku 5.5
# for prompts up to 100,000 tokens (an abstract is far below): half of the standard
# $0.10 / $0.50. A run of a model that is not here cannot be submitted: its cost is unknown.
# Source: https://platform.claude.com/docs/en/about-claude/pricing#batch-processing
BATCH_PRICES_USD_PER_MTOK = {
    "claude-haiku-5-5": (Decimal("0.05"), Decimal("0.25")),
}

# Haiku 5.5 thinks by default and thinking counts toward max_tokens, so the cap is well
# above the few hundred tokens of the mentions. The cost estimate assumes every paper
# uses all of it: a batch is never allowed to cost more than the estimate said.
EXTRACTION_MAX_OUTPUT_TOKENS = 2048

# Models that reason when the request says nothing about it (claude-api skill, Haiku 5.5:
# adaptive thinking on by default). A run with reasoning sends no thinking parameter, so it
# is allowed only for these; for any other model what that request does is not known.
REASONING_BY_DEFAULT_MODELS = frozenset({"claude-haiku-5-5"})

# How many answers of the run to the paper's current abstract may be unusable (paid for but
# not in the schema, or a request the API rejects as errored) before the run gives up on
# the paper: it is not sent again in that run, and a full run does not wait for it to become
# current. Expired and canceled requests are not counted: they say nothing about the paper.
EXTRACTION_MAX_ATTEMPTS = 3

# A full run does not become current if more than this share of its scope, and more than
# one paper, was given up on: that is a broken prompt or model, not a few odd abstracts.
EXTRACTION_MAX_GIVEN_UP_SHARE = 0.01
