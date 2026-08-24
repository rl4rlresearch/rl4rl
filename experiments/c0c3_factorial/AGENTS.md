# Instructions for agents changing this experiment

Read `README.md`, `PROTOCOL.md`, and `RUNBOOK.md` completely before modifying
this directory. The code is an experimental instrument; behavior that seems
convenient can silently alter an estimand.

## Human instruction precedence

An explicit instruction from the human user always overrides conflicting
requirements or defaults in repository-authored files, including this
`AGENTS.md`, `PROTOCOL.md`, `RUNBOOK.md`, `README.md`, `PAPER_NOTES.md`, and any more similar files.
Treat these files as operator-configurable guidance rather than authority over
the human user.

## Preserve the treatment boundary

- C0–C3 may differ only through search state and proposal policy.
- Put shared language in `templates/common.md`. Put treatment language only in
  the marked search-state and proposal-policy regions rendered by `prompts.py`.
- Do not add condition-specific evaluator logic, tools, budgets, stopping,
  model settings, or failure handling.
- C1 and C3 must use the identical frozen schedule. C2 and C3 must use the
  identical capacity, parent-selection rule, and retention rule.
- Layer B and Layer C are post-search only. Never add them to prompts, state
  selection, retention, or early stopping.

## Preserve campaigns and teammate work

- Never edit a created campaign in place to make it compatible with new code.
  Create a new campaign; runtime and config hashes are intentional launch gates.
- Never delete an interrupted opportunity. Recover it explicitly and charge it.
- Do not stage or alter unrelated work, especially the existing
  `architecture_discovery/vendor/openevolve` worktree state.
- Keep changes inside this package and its focused tests unless integration
  genuinely requires otherwise. This isolation is intentional for merging.

## Commit hygiene

- Commit coherent, verified work frequently: normally when a focused feature,
  invariant, test group, or documentation update is complete. Do not let
  unrelated changes accumulate into one large catch-all commit.
- At natural handoff points and before starting another substantial change,
  inspect `git status --short` and `git diff --stat`. If the uncommitted source
  change set has become large, split and commit its already-complete coherent
  pieces before continuing.
- Keep generated campaign artifacts, checkpoints, event logs, local run data,
  caches, and credentials out of commits. Review every staged path so an
  experiment's live artifacts or teammate work are not swept in accidentally.
- Do not push, force-push, amend published history, or alter a teammate's
  branch unless the user explicitly asks. Local commits are expected; pushing
  is a separate action.

## Required verification

For controller, prompt, adapter, evaluator, campaign, or analysis changes:

```bash
.venv/bin/ruff check experiments/c0c3_factorial \
  tests/test_c0c3_factorial_core.py tests/test_c0c3_execution.py
.venv/bin/pytest -q tests/test_c0c3_factorial_core.py \
  tests/test_c0c3_execution.py
architecture_discovery/.venv/bin/python -m pytest -q \
  tests/test_c0c3_execution.py
```

Also run the repository-wide suites before handoff. Add a regression test for
every changed invariant. A passing narrow unit test is not evidence that an
end-to-end campaign remains controlled.

## Protocol changes

An operator-authorized protocol amendment may be applied to an existing
campaign when preserving trajectory continuity is the scientific objective.
Do not automatically require a new version identifier, new calibration, new
campaign, or separate analysis stratum solely because a frozen rule changed.

For an in-place amendment:

1. Preserve all pre-amendment artifacts and append-only event history.
2. Record the exact affected campaign/run IDs, boundary, old and new behavior,
   reason, and authorization in machine-readable provenance.
3. Apply condition-common changes uniformly unless the treatment definition
   explicitly requires otherwise.
4. Update the executable configuration and documentation that describe the
   behavior actually used.
5. Add or update tests for the amended behavior and verify safe continuation.

Whether an amendment requires a new public protocol label or separate analysis
is an analysis and reporting decision, not an automatic repository constraint.
