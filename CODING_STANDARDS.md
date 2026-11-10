# Coding standards

Rules for the reviewer. Each one is a judgement call no automated check can make. Mechanical rules live in tooling instead: ruff lint and format checks on every edited Python file (`.claude/hooks/lint-on-edit.sh`), and the no-AI-attribution check on commit messages and PR descriptions (`scripts/check_no_agent_attribution.sh`, run by the `commit-msg` hook in `.githooks/` and the `attribution` CI job). A finding that tooling already reports is not a review finding.

## Docs state only what the developer decided

Every reason, motive or judgement the README, an ADR, a notebook or a PR body attributes to the developer traces to something the developer said: an issue, a comment, a commit by them, a message in the session. A reason the agent inferred is either cut, or written as the agent's observation with its evidence (a number, a cell, a test).

## Comments say why

A comment explains a reason the code cannot show (a data quirk, a platform limit, a past bug). It does not narrate what the next line does.

## Docs claim what the code does today

A README, an ADR, a spec or an instruction file states as fact only behaviour that exists in the code at that commit. Behaviour that is planned is written as planned, with the issue that will build it. "The caller guarantees", "the hook prevents", "the tool hides" each point at code a reader can find; when the code covers only part of the claim, the doc says which part.
