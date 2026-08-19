#!/usr/bin/env python3
# ruff: noqa: E501
"""Create an isolated, provenance-preserving AdderBoard Autoresearch run.

This command deliberately creates the live experiment under ``data/raw``.
Nothing produced by a run is an input to the analysis package until the run is
over and its raw artifacts have been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTING_MODEL = REPO_ROOT / "architecture_discovery/vendor/starting_model"
ADDERBOARD = REPO_ROOT / "architecture_discovery/vendor/AdderBoard"
RUNNER = REPO_ROOT / "experiments/autoresearch_pilot/run_attempt.py"
TOKEN_TRACKER = REPO_ROOT / "experiments/autoresearch_pilot/token_usage.py"
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
DEFAULT_AGENT_MODEL = "gpt-5.6-terra"
DEFAULT_AGENT_REASONING_EFFORT = "xhigh"


SUBMISSION_WRAPPER = '''"""AdderBoard submission wrapper for this Autoresearch workspace.

The editable model and data implementation live in ``src/``.  This file is
part of the candidate: it must continue to expose the official AdderBoard
``build_model`` and generic autoregressive ``add`` interface.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    FIXED_SEQ_LEN,
    ID_TO_TOKEN,
    OUT_DIGITS,
    VOCAB_SIZE,
    encode,
    postprocess,
    preprocess,
)
from model import AdditionTransformer  # noqa: E402


def _checkpoint_path() -> Path:
    return ROOT / "checkpoints" / "best.pt"


def _build_from_checkpoint() -> tuple[AdditionTransformer, dict]:
    checkpoint = _checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing {checkpoint}. Train the candidate before official verification."
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = AdditionTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        ff_dim=config["ff_dim"],
        max_seq_len=FIXED_SEQ_LEN,
        dropout=0.0,
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def build_model():
    """Return the trained candidate and honest, deduplicated metadata."""
    model, config = _build_from_checkpoint()
    unique_parameters = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "name": "RL4RL Autoresearch candidate",
        "author": "configured in run manifest",
        "params": unique_parameters,
        "architecture": (
            "decoder-only addition transformer "
            f"d={config['d_model']}, h={config['n_heads']}, "
            f"L={config['n_layers']}, ff={config['ff_dim']}"
        ),
        "tricks": ["autoregressive greedy decoding", "weights from checkpoint"],
    }
    return model, metadata


@torch.no_grad()
def add(model, a: int, b: int) -> int:
    """Generic greedy autoregressive decoding; no addition logic lives here."""
    input_ids = [BOS_ID] + encode(preprocess(a, b))
    tokens = torch.tensor([input_ids], dtype=torch.long)
    generated = model.generate(tokens, max_new_tokens=OUT_DIGITS + 1, eos_id=EOS_ID)
    output_ids = generated[0, len(input_ids) :].tolist()
    if EOS_ID in output_ids:
        output_ids = output_ids[: output_ids.index(EOS_ID)]
    raw_output = "".join(ID_TO_TOKEN.get(token, "?") for token in output_ids)
    return postprocess(raw_output)
'''


PROGRAM = """# AdderBoard Autoresearch pilot

You are running a bounded, autonomous research pilot. Your job is to search
for the **smallest trained autoregressive transformer** for 10-digit addition.

## Objective

Minimize the official AdderBoard unique parameter count, subject to at least
99% accuracy on the official verifier (10 edge cases plus 10,000 fixed-seed
random cases, seed 2025). A candidate is retained only when it qualifies and
uses *strictly fewer* parameters than the current retained incumbent.

The starting incumbent is a 6,080-parameter, 100%-verified conventional
decoder-only transformer. Do not assume its architecture is close to globally
optimal. Continue testing substitution-level major architecture changes, not only deletions and
width reductions. Never write a conclusion that a local floor is a global
lower bound.

## Scope and anti-leakage boundary

Work only in this workspace. Do not read the parent RL4RL repository, anything online, or external source code. You may use your own
general technical knowledge, but the experiment should not be handed a known
solution or a target parameter count.

You may edit only:

- `src/model.py`
- `src/data.py`
- `src/train.py`
- `submission.py`
- files under `automations/`, solely to implement bounded automations described
  below

Do not edit `PROGRAM.md`, the official verifier, the runner, `RUN_CONFIG.json`,
or any archived attempt artifacts. Keep the public contracts intact:

- `submission.py` must provide `build_model()` and `add(model, a, b)`;
- the model must remain a tensor-in/logits-out autoregressive transformer with
  at least one self-attention layer;
- `add` must remain generic greedy decoding, with no addition-specific Python
  arithmetic or carry logic;
- the reported parameter count must be calculated from actual, deduplicated
  model parameters. Never manually claim a smaller number.

## Evaluation discipline

During the search, the runner's fixed seed-2025 verifier is a development and
qualification signal, not evidence of final generalization. Do not run it
outside the runner, generate extra test sets to select candidates, or claim
that a retained candidate has independently generalized. A separate final
holdout evaluation is performed only after this run has ended; do not inspect,
generate, or optimize against that final holdout during the search.

## Evidence review before a proposal

Before each regular attempt or automation, inspect the retained source and
`../RESULTS.tsv`, then include all of the following in the `--proposal` text:

1. the current retained parameter count and accuracy margin over 99%;
2. the most recent accepted and failed result in the same mechanism family;
3. the number of prior attempts in that family; and
4. why this proposal is more informative than the nearest untested alternative.

Use one of these mechanism-family labels: `feed-forward width`, `token
representation`, `position representation`, `attention organization`,
`normalization`, `parameter tying`, `scalar pruning`, or `training control`.
The proposal must state its label. A failure is useful evidence; do not repeat
it without explaining what has changed in the hypothesis or conditions.

## Required experiment loop

1. Complete the evidence review and write a concise mechanism hypothesis in
   the `--proposal` and `--description` arguments.
2. Make coherent candidate change(s). A change may include ablations,
   representational substitutions, and local compression ideas.
3. For a regular candidate, run exactly one logged attempt:

   ```bash
   python ../run_attempt.py --run-dir .. attempt --description "short factual description" --proposal "mechanism hypothesis and what changed"
   ```

   The runner trains the current candidate, invokes the untouched official
   verifier, saves code/checkpoint/stdout/stderr before any rollback, appends a
   TSV row, creates a permanent Git ref, and restores the retained incumbent if
   the candidate is not accepted.
4. Read the resulting evidence. Do not relabel a failure as a success and do
   not change the retention rule.
5. Continue until the runner reports that the fixed attempt budget is exhausted,
   then stop and summarize the empirical results without claiming a global
   optimum.

The baseline must be recorded before the first attempt. If it is not already
in `../RESULTS.tsv`, run:

```bash
python ../run_attempt.py --run-dir .. baseline
```

## Non-negotiable completion and serial-execution requirements

This run has a mandatory budget of **__MAX_ATTEMPTS__ macro-attempts**. You
must continue researching until `python ../run_attempt.py --run-dir .. status`
reports `attempts: __MAX_ATTEMPTS__/__MAX_ATTEMPTS__`. Do not stop early
because progress is slow, several ideas have failed, the frontier has not
improved, you think the task is hard, or you cannot immediately think of the
next change. Do not ask the operator whether to continue and do not provide a
final trajectory summary before the budget is exhausted.

Run attempts strictly one at a time. Never start `run_attempt.py` in the
background or from a multi-candidate shell script, never issue a second runner
command while one is active, and never use `kill`, `pkill`, `rmdir`, or `rm` on
runner processes, locks, or attempt artifacts. Empty or unflushed live logs do
not mean training has stalled: the runner's configured timeout is the authority.
If the runner says `Runner busy`, wait for the active command to finish before
making any further source changes or retrying. When an inquiry gives a negative
result, use the evidence to choose the next informative mechanism,
interpolation, training-control condition, or automation. An unrecorded
infrastructure failure is not a reason to end the run.

## Bounded automations

An automation is a small program that executes a repeated, structured
search policy without requiring a separate agent deliberation for every tiny
variant. Use one when testing a monotonic or repeatedly structured mechanism.
The agent's choice of mechanism and search policy is a **macro-attempt**; each
candidate that the program evaluates is a **micro-trial**. If you notice that
your macro-attempts are becoming repetetive, turn it into an automation and
wait until the repetetive task does not work anymore, and use your next macro-
attempt on a different task.

You may create a helper under `automations/` for this purpose. Before launching it,
start the macro-attempt with:

```bash
python ../run_attempt.py --run-dir .. automation-start --automation-id "short-name" --family "scalar pruning" --description "what the automation tests" --proposal "hypothesis, ordering, acceptance, stopping, and budget" --max-micro-trials 20
```

Then have the helper modify one candidate at a time and invoke:

```bash
python ../run_attempt.py --run-dir .. automation-attempt --description "one micro-trial" --proposal "current automation decision"
```

After it reaches its declared boundary, close the macro-attempt with:

```bash
python ../run_attempt.py --run-dir .. automation-end --summary "attempted range, frontier, failures, compute used, and stop reason"
```

Record in the macro proposal:

- the mechanism hypothesis and family label;
- the candidate-ordering rule (for example, an importance ranking);
- the acceptance/rollback rule;
- the stopping rule;
- a maximum micro-trial count; and
- a training or wall-clock compute budget.

The automation must modify one candidate at a time and invoke
`automation-attempt` once
for every micro-trial. It must never train, verify, retain, or silently discard
candidates outside `run_attempt.py`. `AUTOMATION_RESULTS.tsv` records every
micro-trial; `RESULTS.tsv` receives one summary row when `automation-end` closes
the macro-attempt. After each micro-trial, the helper may read
`../AUTOMATION_RESULTS.tsv`
and `../STATE.json` to decide its next eligible candidate. Stop the automation at its declared boundary, then write the required
`automation-end` summary. Only then should you select a new mechanism or revise
the automation.

Do **not** create a series of automation IDs with
`--max-micro-trials 1` to test a repeated list one candidate at a time. A
one-micro automation is allowed only for a genuinely non-repeatable research
question, and its proposal must explain why no ordered multi-trial helper is
valid. The goal with the micro trials and automations is to save tokens and
macro-attempts. The maximum micro-trial count, and any other limits on
automations should allow the automation to keep going until it is exhausted
rather than hitting a limit early without exhausting itself. Do not keep
checking on your automations every micro-trial, it is preffered that you to
trust the automation that you ran will work and set triggers to let you know
of any problems or when the automation is completed. It is also preffered
that you sit idle while the automation runs, only acting if it is completed
or there is an error/problem that requires your action.

An automation may stop early only because it reached its declared cap, no
eligible candidate remains, a scored failure changes the declared eligibility
rule, or it encounters a reproducible error. Record the concrete stop reason
in `automation-end`. Never terminate a live runner based on empty or buffered
logs, and never run runner commands concurrently; wait for the active command
or its configured timeout.

### Autonomous execution and triggers

Launch an automation helper once as a foreground, blocking command, then leave
it alone until it exits. Do not tail, poll, pause, or send a progress message
after a fixed number of successful micro-trials. A qualifying micro-trial is
normal operation, **not** a trigger. Keep helpers quiet while they are making
progress: do not print one line per successful micro-trial.

The same rule applies to ordinary attempts: once launched, do not poll or send
progress messages until the runner returns, times out, or produces an error.

The only reasons for the helper to return control are: a scored `discard`, a
recorded `error`, the declared cap, no eligible candidate, a runner nonzero
exit or timeout, or another reproducible infrastructure problem. The runner
writes `AUTOMATION_TRIGGER.json` in the active automation directory for a
scored discard/error or a reached cap. A helper must write the same JSON file
for no-candidate and runner-failure triggers, containing the reason, the last
micro-trial ID if one exists, and a concise detail. When the helper exits,
read that one trigger artifact and the final recorded result, call
`automation-end` exactly once, and then decide what to do next. Do not create
routine agent messages merely to report that the automation is working.

## Interpreting errors

When an attempt has status `error`, inspect its train and verifier logs before
making another proposal. Classify it in the next proposal as one of:

- `infrastructure`: command, timeout, dependency, or filesystem problem;
- `implementation`: broken model, data, or submission contract;
- `optimization failure`: the candidate trained but never reached the
  checkpoint/validation criterion; or
- `nonqualification`: a valid checkpoint received a score below threshold.

An error without an official score is not automatically evidence that the
architecture lacks capacity. You may repair one reproducible infrastructure or
implementation error for the same intended candidate, but must identify it as
a repair. Do not silently retry an optimization failure; explain why changed
conditions justify it.

## What to preserve in reasoning

For each proposal, distinguish a parameterization-preserving compression from
a representational change (for example, token representation, positional
integration, deterministic/tied projections, attention organization, or
feed-forward mechanism). The source snapshots and Codex event log are research
artifacts, not scratch files.

## Final report only

Only after the configured attempt budget is exhausted, provide a trajectory
summary in the final response. Do not write this full summary after each
attempt. State the initial and final frontier, number of regular attempts and
micro-trials, allocation across mechanism families, accepted/error/
nonqualification outcomes, important failure boundaries, and the training
steps or wall-clock cost where the logs provide them. State that independent
final-holdout evaluation remains required, and do not claim a global optimum.
"""


def _program(max_attempts: int) -> str:
    return PROGRAM.replace("__MAX_ATTEMPTS__", str(max_attempts))


AGENT_PROMPT = """Read `PROGRAM.md` and operate as the autonomous researcher described there.

First confirm that the baseline has been recorded. Then perform logged
experiments until the run's configured attempt budget is exhausted. Do not ask
the operator to choose research ideas; make and test your own bounded,
well-documented choices. Follow every scope, anti-leakage, integrity, and
rollback requirement in `PROGRAM.md`. Your final response must state the run
directory and the number of attempts actually recorded, without claiming that
the observed frontier is a global optimum. Do not end the session before the
configured attempt budget is exhausted; the launcher will relaunch an
early-stopped session until that budget is reached.
"""


LAUNCH_SCRIPT = """#!/usr/bin/env bash
set -uo pipefail

CODEX_MODEL=__CODEX_MODEL__
CODEX_REASONING_EFFORT=__CODEX_REASONING_EFFORT__
PYTHON_BIN=__PYTHON_BIN__
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$RUN_DIR/workspace"
TOKEN_TRACKER="$RUN_DIR/token_usage.py"
mkdir -p "$RUN_DIR/logs"
LAUNCH_LOCK="$RUN_DIR/.launcher.lock"

if ! mkdir "$LAUNCH_LOCK" 2>/dev/null; then
  owner="$(cat "$LAUNCH_LOCK/pid" 2>/dev/null || true)"
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    echo "A launcher is already active for this run (pid $owner)." >&2
    exit 3
  fi
  rm -rf "$LAUNCH_LOCK"
  mkdir "$LAUNCH_LOCK"
fi
printf '%s\n' "$$" > "$LAUNCH_LOCK/pid"
trap 'rm -rf "$LAUNCH_LOCK"' EXIT INT TERM

read_json_int() {
  "$PYTHON_BIN" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' "$1" "$2"
}

MAX_ATTEMPTS="$(read_json_int "$RUN_DIR/RUN_CONFIG.json" max_attempts)"
session=0

while [ "$(read_json_int "$RUN_DIR/STATE.json" attempts_used)" -lt "$MAX_ATTEMPTS" ]; do
  session=$((session + 1))
  printf '%s session=%s attempts=%s/%s starting Codex\n' "$(date -u +%FT%TZ)" "$session" "$(read_json_int "$RUN_DIR/STATE.json" attempts_used)" "$MAX_ATTEMPTS" >> "$RUN_DIR/logs/supervisor.log"
  codex exec --model "$CODEX_MODEL" \\
    -c "model_reasoning_effort=$CODEX_REASONING_EFFORT" \\
    --json --approve-for-me \\
    --cd "$WORKSPACE" --add-dir "$RUN_DIR" \\
    --output-last-message "$RUN_DIR/logs/codex-last-message.md" \\
    "$(cat "$RUN_DIR/AGENT_PROMPT.md")" \\
    | tee -a "$RUN_DIR/logs/codex-events.jsonl"
  codex_exit=${PIPESTATUS[0]}
  "$PYTHON_BIN" "$TOKEN_TRACKER" --runs-root "$RUN_DIR/.." --output-dir "$RUN_DIR/../token_usage" >/dev/null 2>&1 || \
    printf '%s session=%s token accounting refresh failed\n' "$(date -u +%FT%TZ)" "$session" >> "$RUN_DIR/logs/supervisor.log"
  used="$(read_json_int "$RUN_DIR/STATE.json" attempts_used)"
  if [ "$used" -lt "$MAX_ATTEMPTS" ]; then
    printf '%s session=%s exited=%s before budget (%s/%s); relaunching in 10s\n' "$(date -u +%FT%TZ)" "$session" "$codex_exit" "$used" "$MAX_ATTEMPTS" >> "$RUN_DIR/logs/supervisor.log"
    sleep 10
  fi
done
printf '%s budget exhausted at %s/%s; supervisor exiting\n' "$(date -u +%FT%TZ)" "$(read_json_int "$RUN_DIR/STATE.json" attempts_used)" "$MAX_ATTEMPTS" >> "$RUN_DIR/logs/supervisor.log"
"""


def _launch_script(model: str, reasoning_effort: str, python_bin: str) -> str:
    """Render a shell-safe launcher with the run's explicit model settings."""
    return (
        LAUNCH_SCRIPT.replace("__CODEX_MODEL__", shlex.quote(model))
        .replace("__CODEX_REASONING_EFFORT__", shlex.quote(reasoning_effort))
        .replace("__PYTHON_BIN__", shlex.quote(str(python_bin)))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(directory: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True
    ).strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _copy_source(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".DS_Store", "*.pyc"),
    )


def _init_workspace_git(workspace: Path) -> str:
    commands = (
        ("init", "-b", "main"),
        ("config", "user.name", "RL4RL Autoresearch"),
        ("config", "user.email", "autoresearch@rl4rl.local"),
        ("add", "src", "submission.py", "PROGRAM.md", ".gitignore"),
        ("commit", "-m", "baseline: 6080-parameter starting candidate"),
    )
    for command in commands:
        subprocess.run(["git", "-C", str(workspace), *command], check=True)
    return _git_output(workspace, "rev-parse", "HEAD")


def create_run(args: argparse.Namespace) -> Path:
    if not RUN_ID_RE.fullmatch(args.run_id):
        raise ValueError(
            "run id must be 3-64 lowercase letters, digits, and hyphens "
            "(for example: autoresearch-pilot-20260814)"
        )
    if not (STARTING_MODEL / "src" / "model.py").is_file():
        raise FileNotFoundError(
            "starting-model submodule is unavailable; run "
            "git submodule update --init architecture_discovery/vendor/starting_model"
        )
    if not (ADDERBOARD / "verify.py").is_file():
        raise FileNotFoundError(
            "AdderBoard submodule is unavailable; run "
            "git submodule update --init architecture_discovery/vendor/AdderBoard"
        )
    if not RUNNER.is_file():
        raise FileNotFoundError(f"Missing runner: {RUNNER}")
    if not TOKEN_TRACKER.is_file():
        raise FileNotFoundError(f"Missing token tracker: {TOKEN_TRACKER}")

    runs_root = Path(args.runs_root).expanduser().resolve()
    run_dir = runs_root / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")

    workspace = run_dir / "workspace"
    run_runner = run_dir / "run_attempt.py"
    run_dir.mkdir(parents=True)
    try:
        _copy_source(STARTING_MODEL / "src", workspace / "src")
        (workspace / "checkpoints").mkdir(parents=True)
        (workspace / "automations").mkdir()
        shutil.copy2(
            STARTING_MODEL / "checkpoints" / "best.pt",
            workspace / "checkpoints" / "best.pt",
        )
        (workspace / "submission.py").write_text(SUBMISSION_WRAPPER, encoding="utf-8")
        (workspace / "PROGRAM.md").write_text(
            _program(args.max_attempts), encoding="utf-8"
        )
        (workspace / ".gitignore").write_text(
            "checkpoints/\nresults/\n__pycache__/\n*.py[cod]\n",
            encoding="utf-8",
        )

        shutil.copy2(ADDERBOARD / "verify.py", run_dir / "official_verify.py")
        shutil.copy2(RUNNER, run_runner)
        shutil.copy2(TOKEN_TRACKER, run_dir / "token_usage.py")
        (run_dir / "attempts").mkdir()
        (run_dir / "state").mkdir()
        (run_dir / "logs").mkdir()
        (run_dir / "AGENT_PROMPT.md").write_text(AGENT_PROMPT, encoding="utf-8")
        (run_dir / "launch_codex.sh").write_text(
            _launch_script(
                args.agent_model, args.agent_reasoning_effort, args.python_bin
            ),
            encoding="utf-8",
        )
        (run_dir / "launch_codex.sh").chmod(0o755)

        baseline_commit = _init_workspace_git(workspace)
        state = {
            "schema": "rl4rl-autoresearch-state-v1",
            "baseline_recorded": False,
            "attempts_used": 0,
            "incumbent": {
                "attempt_id": "baseline",
                "commit": baseline_commit,
                "parameters": None,
                "checkpoint": "state/incumbent.pt",
            },
        }
        _write_json(run_dir / "STATE.json", state)
        (run_dir / "RESULTS.tsv").write_text(
            "attempt_id\tcommit\tparent_commit\ttimestamp_utc\taccuracy\tparameters"
            "\tstatus\tdescription\tproposal\ttrain_exit\tverify_exit\n",
            encoding="utf-8",
        )
        (run_dir / "AUTOMATION_RESULTS.tsv").write_text(
            "macro_attempt_id\tautomation_id\tmicro_attempt_id\tcommit\tparent_commit"
            "\ttimestamp_utc\tfamily\taccuracy\tparameters\tstatus\tdescription"
            "\tproposal\ttrain_exit\tverify_exit\n",
            encoding="utf-8",
        )

        inputs = {
            "starting_model/src/model.py": STARTING_MODEL / "src" / "model.py",
            "starting_model/src/data.py": STARTING_MODEL / "src" / "data.py",
            "starting_model/src/train.py": STARTING_MODEL / "src" / "train.py",
            "starting_model/checkpoints/best.pt": STARTING_MODEL
            / "checkpoints"
            / "best.pt",
            "adderboard/verify.py": ADDERBOARD / "verify.py",
            "pilot/create_run.py": Path(__file__).resolve(),
            "pilot/run_attempt.py": RUNNER,
            "pilot/token_usage.py": TOKEN_TRACKER,
        }
        manifest = {
            "schema": "rl4rl-autoresearch-run-v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_id": args.run_id,
            "objective": "minimize qualified AdderBoard unique parameter count",
            "qualification": {
                "minimum_accuracy": 0.99,
                "random_cases": 10000,
                "random_seed": 2025,
                "edge_cases": 10,
            },
            "retention_rule": "qualifies and strictly fewer parameters than incumbent",
            "max_attempts": args.max_attempts,
            "author": args.author,
            "operator_note": args.note,
            "agent_model": args.agent_model,
            "agent_reasoning_effort": args.agent_reasoning_effort,
            "python_bin": str(Path(args.python_bin).expanduser()),
            "workspace_baseline_commit": baseline_commit,
            "source_revisions": {
                "adderboard": _git_output(ADDERBOARD, "rev-parse", "HEAD"),
                "starting_model": _git_output(STARTING_MODEL, "rev-parse", "HEAD"),
                "rl4rl": _git_output(REPO_ROOT, "rev-parse", "HEAD"),
            },
            "input_sha256": {name: _sha256(path) for name, path in inputs.items()},
        }
        _write_json(run_dir / "RUN_MANIFEST.json", manifest)
        _write_json(
            run_dir / "RUN_CONFIG.json",
            {
                "schema": "rl4rl-autoresearch-config-v1",
                "max_attempts": args.max_attempts,
                "agent_model": args.agent_model,
                "agent_reasoning_effort": args.agent_reasoning_effort,
                "python_bin": manifest["python_bin"],
                "runner": str(run_runner),
                "training_command": [manifest["python_bin"], "src/train.py"],
                "training_timeout_seconds": args.training_timeout,
                "verification_timeout_seconds": args.verification_timeout,
            },
        )
    except BaseException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", required=True, help="immutable raw-artifact directory name"
    )
    parser.add_argument(
        "--runs-root",
        default=REPO_ROOT / "data/raw/autoresearch",
        help="parent directory for raw runs (default: data/raw/autoresearch)",
    )
    parser.add_argument(
        "--author", default="unknown", help="operator/author recorded in manifest"
    )
    parser.add_argument(
        "--note", default="", help="short operator note recorded in manifest"
    )
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument(
        "--agent-model",
        default=DEFAULT_AGENT_MODEL,
        help=f"Codex model for the researcher (default: {DEFAULT_AGENT_MODEL})",
    )
    parser.add_argument(
        "--agent-reasoning-effort",
        default=DEFAULT_AGENT_REASONING_EFFORT,
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        help=(
            "Codex reasoning effort for the researcher "
            f"(default: {DEFAULT_AGENT_REASONING_EFFORT})"
        ),
    )
    parser.add_argument(
        "--python-bin",
        default=REPO_ROOT / "architecture_discovery/.venv/bin/python",
        help="Python executable with PyTorch installed",
    )
    parser.add_argument("--training-timeout", type=int, default=3600)
    parser.add_argument("--verification-timeout", type=int, default=900)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="create an unverified run without baseline evaluation (testing/recovery only)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")
    if args.training_timeout < 1 or args.verification_timeout < 1:
        raise SystemExit("timeouts must be positive")
    run_dir = create_run(args)
    print(run_dir)
    if not args.skip_baseline:
        print("Recording the official baseline before the run is ready...", flush=True)
        baseline = subprocess.run(
            [
                str(args.python_bin),
                str(run_dir / "run_attempt.py"),
                "--run-dir",
                str(run_dir),
                "baseline",
            ],
            check=False,
        )
        baseline_state = json.loads((run_dir / "STATE.json").read_text(encoding="utf-8"))
        if baseline.returncode != 0 or not baseline_state["baseline_recorded"]:
            print(
                "Baseline failed; the run and its diagnostic artifacts were retained at "
                f"{run_dir}.",
                file=sys.stderr,
            )
            return baseline.returncode or 1
    print(
        f"Baseline recorded. Next: {run_dir / 'launch_codex.sh'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
