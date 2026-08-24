# Completion audit

Audit date: 2026-08-20. This file maps the original implementation request to
authoritative evidence. It is not a substitute for each campaign’s launch-time
`validation.json`.

## Requirement evidence

| Requirement | Evidence | Result |
|---|---|---|
| New branch from `main`, no push | Branch `codex/c0c3-codex-factorial`; merge base equals local `main` commit `4517f548`; incremental commits; no remote push performed | Satisfied |
| Exact C0–C3 2×2 mapping | `spec.Condition`, `test_condition_mapping_and_frozen_transition_schedule` | Satisfied |
| C0/C1 single incumbent | `SearchController.begin/complete`; strict-improvement test | Satisfied |
| C2/C3 portfolio memory | Frozen `K`, seed-fill, lineage-fair selection, replacement logic; fill/fairness regression tests | Satisfied |
| Same C1/C3 transition schedule | One `transition_opportunities` tuple and prompt-hash pairing audit over every opportunity | Satisfied |
| Frozen capacity/retention/selection/failure/order | Versioned rule identifiers in `spec.py`, strict TOML parser, paper protocol, behavior tests | Satisfied |
| Controlled prompt structure | One `common.md`, two marked treatment regions, neutral K slots, byte-redacted skeleton tests and launch audit | Satisfied |
| Identical model/task/tools/Layer A/budgets within cells | One campaign-level immutable spec/task/framework; hashes copied to every manifest; scientific-runtime gate before every opportunity | Satisfied |
| No online Layer B/C | Common prompt prohibition, no controller dependency on postsearch modules, export/evaluation sealed until all runs complete, tests | Satisfied |
| Required per-proposal logging | `events.jsonl` records condition, visible/selected IDs, type, hypothesis/edit, lineage, evaluation, retention, tokens, evaluator calls/time, and remaining budget; field test | Satisfied |
| Memory/policy/interaction estimates | Balanced/duplicate-complete checks, cell means, exact contrasts, within-block contrasts, estimator tests | Satisfied |
| Primary mechanism-cluster outcome | Blinded parent/candidate packets, hidden condition/scores, annotation validation, distinct per-run cluster scoring, two-reviewer protocol in docs | Satisfied |
| Separate no-search baseline | N0 always starts at seed, receives no adaptive feedback/transitions, excluded from contrasts; offline best-of-independent Layer C selection; tests | Satisfied |
| Karpathy Autoresearch and OpenEvolve | Direct-workspace adapter and controlled vendored OpenEvolve prompt/diff adapter; fake-Codex end-to-end tests in both dependency environments | Satisfied |
| Easy future framework/task changes | Strict independent TOMLs, `FrameworkSpec`/`TaskSpec`, narrow adapter factory/evaluator contract, extension instructions | Satisfied |
| AdderBoard ready | Real starting model/support packaging, trusted verifier wrapper, 99% qualification, parameter objective, fixed Layer A and disjoint Layer C seeds | Satisfied |
| At least one other strong ML task | Pinned-source official Karpathy Autoresearch nanoGPT config, H100 fixed-time `val_bpb`, persistent preparation cache, target-backend calibration path | Satisfied |
| Local Codex CLI | Official noninteractive flags, ephemeral calls, JSONL/last-message capture, usage parser, run and campaign locks/recovery, serial and parallel campaign CLIs | Satisfied |
| Versioned parallel rounds | Protocol 1.1 binds a distinct execution-rule identifier, concurrently launches least-advanced C0–C3 peers behind a start barrier, serializes N0, logs every wave, and deterministically selects lagging recovery subsets; concurrency/recovery/lock tests | Satisfied |
| Modal parity | Protocol 1.0 uses the same CLI/controller on pinned Modal SDK, H100 image, secret, campaign/cache Volumes, explicit reload/commit, no retries, one mutation container, portable calibration; validation explicitly rejects protocol 1.1 on nonlocal task backends | Satisfied |
| Merge safety and reuse | New isolated package; reuses vendored OpenEvolve and AdderBoard; no edits/staging of teammate-owned dirty submodule | Satisfied |
| Detailed human/agent docs | README, protocol, runbook, framework/task guide, Modal guide, agent instructions, paper notes, and this audit | Satisfied |
| Literature-informed decisions | FML-bench, Heuresis, Autoresearch, OpenEvolve, EvoTrace, and long-horizon architecture work synthesized with links in `PAPER_NOTES.md` | Satisfied |
| Tests for additions | Core unit tests, transport/evaluator/campaign/postsearch/Modal path integrations, real dependency-environment test, real source packaging smoke | Satisfied |

## Final verification evidence

Passed after the final implementation changes:

```text
.venv/bin/pytest -q
  PASS (root suite; one expected optional-dependency skip)

architecture_discovery/.venv/bin/python -m pytest -q tests/test_c0c3_execution.py
  PASS (14 tests, including actual vendored OpenEvolve dependencies, Modal SDK
  import, synchronized four-call execution, and campaign-lock rejection)

.venv/bin/ruff check experiments/c0c3_factorial \
  tests/test_c0c3_factorial_core.py tests/test_c0c3_execution.py
  PASS

git diff --check
  PASS

PYTHONPATH=. architecture_discovery/.venv/bin/modal run \
  -m experiments.c0c3_factorial.modal_app --help
  PASS; all generated options load under pinned modal==1.5.3
```

Additional real-source smokes:

- A fresh official Karpathy Autoresearch checkout at commit
  `228791fb499afffb54b46200aca536f79142f117` produced a portable calibration
  bundle with `train.py`/`prepare.py`, no `.git`, and no fabricated baseline.
- The real AdderBoard starting tree produced a portable bundle containing its
  training source and trusted submission wrapper.
- All two tasks × two framework configs parsed and generated runtime hashes.
- The controlled OpenEvolve SEARCH/REPLACE adapter passed against both the
  shared working checkout and the clean parent-repository-pinned commit
  `5ecb48b5ca453d3f2b9c316a4ffe45d45725bf0c`.

No provider-token or H100 training call was made during implementation. Actual
calibration is intentionally an operator launch step because it spends compute
and must bind the chosen final task commit/backend. The commands and fail-closed
gates for doing so are in `RUNBOOK.md` and `MODAL.md`.

## Shared-workspace exception preserved

The pre-existing `architecture_discovery/vendor/openevolve` submodule is dirty
in the parent worktree: the parent pins `5ecb48b5...`, while the shared nested
checkout is at `258c29b...`. The full architecture-discovery suite also scans
large pre-existing `data/raw` experiment artifacts and exceeds that subsystem’s
manifest file limit. Consequently its broad state-dependent suite reports many
downstream patch-bundle/readiness failures in this shared workspace.

This branch did not alter or stage that submodule or raw run data. The focused
C0–C3 integration suite passes against the shared checkout, and the clean pinned
OpenEvolve adapter smoke above proves merge/fresh-clone compatibility. Resetting
the teammate-owned checkout or deleting raw artifacts merely to make unrelated
stateful tests green would violate the merge-safety requirement.

## Launch boundary

Implementation is complete; paper data collection is not. Before the first real
opportunity, the operator must still:

1. choose and record the exact official Autoresearch commit;
2. authenticate Codex and Modal and confirm budgets;
3. execute calibration on the final backend;
4. create each campaign from the final committed source;
5. run its launch validation and archive the environment receipt;
6. keep exactly one mutation writer per campaign.

Those are experimental operations, not missing implementation. A campaign is
not valid merely because this package passed its engineering audit; its own
launch receipt remains mandatory.
