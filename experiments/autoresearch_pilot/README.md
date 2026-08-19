# Local AdderBoard Autoresearch pilot

This is the runnable first experiment for the original RL4RL research question:
how an autonomous researcher explores (or fails to explore) architectural
alternatives while minimizing AdderBoard's qualified parameter count. It is a
small, bounded **pilot**, not a 30-day replication or evidence of a general
system effect.

The scaffold starts with the verified 6,080-parameter model. It creates an
isolated Git workspace under `data/raw/autoresearch/<run-id>/`, exposes an
official AdderBoard submission wrapper, and records every candidate before it
is accepted or rolled back. The agent is never shown the public leaderboard or
the known 36-parameter trained frontier.

## Before you start

1. Work from a clean top-level repository checkout. The run is stored under
   `data/raw`, which is intentionally Git-ignored, so it cannot accidentally
   be committed with analysis code.
2. Confirm the two required submodules exist:

   ```bash
   git submodule update --init architecture_discovery/vendor/AdderBoard \
     architecture_discovery/vendor/starting_model
   ```

3. Use a Python executable with PyTorch. On this computer the existing
   `architecture_discovery/.venv/bin/python` already has it. On a fresh machine
   create that environment with:

   ```bash
   uv sync --project architecture_discovery
   ```

   Do **not** add PyTorch to the root analysis package just to run this pilot.

## Create the run

Choose a stable, descriptive ID and a modest attempt cap. Five full training
attempts is a useful first pilot: it produces a real trajectory without
pretending to replicate the inherited 30-day run.

```bash
architecture_discovery/.venv/bin/python \
  experiments/autoresearch_pilot/create_run.py \
  --run-id autoresearch-pilot-20260814 \
  --author "your-name" \
  --note "first independent bounded pilot" \
  --max-attempts 5 \
  --agent-model gpt-5.6-terra \
  --agent-reasoning-effort xhigh
```

More concise version:

```bash
architecture_discovery/.venv/bin/python \
  experiments/autoresearch_pilot/create_run.py \
  --run-id autoresearch-pilot-20260814 \
  --max-attempts 5 \
  --agent-model gpt-5.6-terra \
  --agent-reasoning-effort xhigh
```

The command prints the absolute run directory and automatically evaluates the
unmodified 6,080-parameter model with the official seed-2025 verifier. It does
not call an LLM or train anything. A successful command leaves a qualified
`baseline` row in `<RUN_DIR>/RESULTS.tsv`; if qualification fails, it exits
nonzero but preserves the new run and its diagnostics for inspection.

The model and reasoning effort are chosen when the run is created and are
recorded in the run manifest, configuration, and launcher. If omitted, they
default to `gpt-5.6-terra` with `xhigh` reasoning. For example, to use a
different model, add `--agent-model <model-name>`; to change reasoning, use
`--agent-reasoning-effort high` (the accepted values are `minimal`, `low`,
`medium`, `high`, and `xhigh`). These settings do not inherit from your global
Codex `config.toml`.

## Launch the autonomous researcher

First inspect these run-local files:

- `<RUN_DIR>/RUN_MANIFEST.json`: frozen source revisions, hashes, seed,
  retention rule, author, and run budget.
- `<RUN_DIR>/workspace/PROGRAM.md`: the exact research instructions available
  to the agent.
- `<RUN_DIR>/RUN_CONFIG.json`: the Python executable and hard timeouts.
- `<RUN_DIR>/RESULTS.tsv`: confirm the automatically recorded `baseline` row
  qualified before launching.

Then launch the non-interactive Codex run:

```bash
<RUN_DIR>/launch_codex.sh
```

This is the step that consumes your configured Codex/API allowance. It stores
the raw Codex event stream in `<RUN_DIR>/logs/codex-events.jsonl` and its final
message in `logs/codex-last-message.md`. Do not use a separate interactive
agent in the same workspace while it is running.

## Token accounting

Token accounting is kept outside all run directories, so it cannot affect a
live workspace or runner. To generate totals for every completed and ongoing
run, plus each model-response increment, run:

```bash
architecture_discovery/.venv/bin/python \
  experiments/autoresearch_pilot/token_usage.py \
  --runs-root data/raw/autoresearch
```

This writes `run_totals.tsv`, `session_totals.tsv`, and
`response_increments.tsv` to `data/raw/autoresearch/token_usage/`. The first
two use Codex's cumulative session totals; forked copies of the same Codex
session are deduplicated by logical session ID. `response_increments.tsv` records
the per-response `last_token_usage` increments. Cached input is reported
separately and is already included in input-token totals. Add `--watch-seconds
60` for a read-only live refresh. Future launchers refresh the same reports
after each Codex session exits.

Mutating runner commands now hold an exclusive `.runner.lock` for their full
duration. A second command targeting the same run exits with `Runner busy`
instead of reusing an attempt ID or corrupting `STATE.json`. Different run
directories, such as separate Sol and Luna runs, can still run concurrently.

The agent must use the runner once per candidate. The runner does all of the
following before restoring a rejected change:

1. saves the proposed source snapshot and a permanent Git ref;
2. trains the current candidate from `workspace/src/train.py`;
3. saves training stdout/stderr and the resulting checkpoint;
4. invokes the untouched official verifier with seed 2025;
5. records the accuracy, unique parameter count, parent commit, hypothesis,
   exit codes, and retention outcome in `RESULTS.tsv`; and
6. keeps the candidate only if it passes and is strictly smaller; otherwise it
   restores the previous source and checkpoint.

The candidate source may change `src/model.py`, `src/data.py`, `src/train.py`,
and `submission.py`. The official verifier and runner are outside the
workspace, making accidental modification of the measuring system harder.

## Monitor and stop safely

In a second terminal:

```bash
tail -f <RUN_DIR>/RESULTS.tsv
architecture_discovery/.venv/bin/python \
  <RUN_DIR>/run_attempt.py --run-dir <RUN_DIR> status
```

The runner stops accepting commands after `max_attempts`; a baseline does not
count toward that limit. If you need to halt the agent, interrupt only the
Codex process (`Ctrl-C`). Do not delete files or run `git gc` inside the
workspace: the unaccepted attempt refs are part of the trajectory evidence.

If you stop after a completed attempt, the retained incumbent is already safe.
If you interrupt during training, keep the partial attempt directory and note
the interruption in a new operator note; it is still evidence of resource use,
not a missing row to erase.

## Inspect and ingest the completed pilot

The raw source of truth is the run directory. In particular:

```text
RUN_MANIFEST.json        frozen inputs and hashes
RUN_CONFIG.json          command/time budget
STATE.json               current retained incumbent
RESULTS.tsv              one row per baseline/attempt
AUTOMATION_RESULTS.tsv   one row per automation micro-trial
attempts/<id>/           source, checkpoint, stdout, stderr, parsed result
automations/<macro-id>/  automation plan, micro-trial artifacts, and final summary
workspace/.git/          retained line plus permanent rejected-attempt refs
logs/codex-events.jsonl  agent reasoning/action event stream
```

For a bounded automation, reserve one macro attempt, run one or more
micro-trials, then close it. `RESULTS.tsv` receives the one macro summary;
`AUTOMATION_RESULTS.tsv` receives every candidate-level record:

```bash
<RUN_DIR>/run_attempt.py --run-dir <RUN_DIR> automation-start \
  --automation-id qkv-prune \
  --family "scalar pruning" \
  --description "greedy QKV scalar-pruning automation" \
  --proposal "hypothesis, ordering, acceptance, stopping, and budget" \
  --max-micro-trials 20

<RUN_DIR>/run_attempt.py --run-dir <RUN_DIR> automation-attempt \
  --description "zero the next eligible scalar" \
  --proposal "ranked candidate 1"

<RUN_DIR>/run_attempt.py --run-dir <RUN_DIR> automation-end \
  --summary "boundary reached after 12 micro-trials"
```

After the run ends, inspect the baseline and at least ten candidate artifacts
(or every candidate for this small pilot). Then normalize the TSV:

```bash
uv run rl4rl parse-autoresearch <RUN_DIR>/RESULTS.tsv \
  --run-id autoresearch-pilot-20260814 \
  --output data/interim/autoresearch-pilot-20260814.jsonl

uv run rl4rl validate data/interim/autoresearch-pilot-20260814.jsonl
```

The normalized JSONL is not yet an experimental result. Before reporting
rates, independently annotate each edit as representationally preserving,
crossing, or ambiguous using `configs/taxonomy.toml`, and distinguish a short
pilot from independent replications.

## Important interpretation limits

- Five attempts measure whether the instrumentation works; they cannot establish
  premature convergence or compare systems.
- The public frontier is intentionally withheld from the agent, but an LLM may
  still have related knowledge from pretraining. Describe this as a controlled
  input restriction, not proof of no prior exposure.
- Parameter count is the optimization objective for this reproduction because
  that is the original pasted protocol. This intentionally differs from the
  separate `architecture_discovery/` branch protocol, where parameter count is
  descriptive metadata only.
