# Instructions for agents changing this experiment

Read `README.md`, `PROTOCOL.md`, and `RUNBOOK.md` completely before modifying
this directory. The code is an experimental instrument; behavior that seems
convenient can silently alter an estimand.

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

If a frozen rule must change:

1. Change its versioned identifier in `spec.py`.
2. Update every protocol TOML explicitly.
3. Update `PROTOCOL.md` and the protocol-deviation log in `PAPER_NOTES.md`.
4. Add or update tests that distinguish the old and new behavior.
5. Never combine runs from the two versions without labeling them separately.
