"""Fixed choices of the project that code in several modules needs."""

from datetime import date

# Slices that are collected. Adding one here is all the collector needs; the cost of
# extracting the new papers is a separate decision (CLAUDE.md).
CATEGORIES = ("cs.CL",)

# The history starts here and only grows; nothing older is stored (docs/adr/0003).
HISTORY_START = date(2025, 11, 1)
