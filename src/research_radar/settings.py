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

# How many collected batches may carry a paper, in one run and with the same abstract,
# without a usable answer before the run gives up on it. A full run does not wait for such
# a paper to become current, and the daily job does not send it again.
EXTRACTION_MAX_ATTEMPTS = 3
