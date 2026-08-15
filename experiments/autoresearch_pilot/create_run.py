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
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTING_MODEL = REPO_ROOT / "architecture_discovery/vendor/starting_model"
ADDERBOARD = REPO_ROOT / "architecture_discovery/vendor/AdderBoard"
RUNNER = REPO_ROOT / "experiments/autoresearch_pilot/run_attempt.py"
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
for a **smaller trained autoregressive transformer** for 10-digit addition.

## Objective

Minimize the official AdderBoard unique parameter count, subject to at least
99% accuracy on the official verifier (10 edge cases plus 10,000 fixed-seed
random cases, seed 2025). A candidate is retained only when it qualifies and
uses *strictly fewer* parameters than the current retained incumbent.

The starting incumbent is a 6,080-parameter, 100%-verified conventional
decoder-only transformer. Do not assume its architecture is close to globally
optimal. Continue testing substitution-level changes, not only deletions and
width reductions. Never write a conclusion that a local floor is a global
lower bound.

## Scope and anti-leakage boundary

Work only in this workspace. Do not read the parent RL4RL repository, online
leaderboards, other submissions, or external source code. You may use your own
general technical knowledge, but the experiment should not be handed a known
solution or a target parameter count.

You may edit only:

- `src/model.py`
- `src/data.py`
- `src/train.py`
- `submission.py`

Do not edit `PROGRAM.md`, the official verifier, the runner, `RUN_CONFIG.json`,
or any archived attempt artifacts. Keep the public contracts intact:

- `submission.py` must provide `build_model()` and `add(model, a, b)`;
- the model must remain a tensor-in/logits-out autoregressive transformer with
  at least one self-attention layer;
- `add` must remain generic greedy decoding, with no addition-specific Python
  arithmetic or carry logic;
- the reported parameter count must be calculated from actual, deduplicated
  model parameters. Never manually claim a smaller number.

## Required experiment loop

1. Inspect the retained source and `../RESULTS.tsv`.
2. Write a concise mechanism hypothesis and proposal in the `--proposal` and
   `--description` arguments below.
3. Make one coherent candidate change. A change may be an ablation, but include
   representational substitutions as well as local compression ideas.
4. Run exactly one logged attempt:

   ```bash
   python ../run_attempt.py --run-dir .. attempt --description "short factual description" --proposal "mechanism hypothesis and what changed"
   ```

   The runner trains the current candidate, invokes the untouched official
   verifier, saves code/checkpoint/stdout/stderr before any rollback, appends a
   TSV row, creates a permanent Git ref, and restores the retained incumbent if
   the candidate is not accepted.
5. Read the resulting evidence. Do not relabel a failure as a success and do
   not change the retention rule.
6. Continue until the runner reports that the fixed attempt budget is exhausted,
   then stop and summarize the empirical results without claiming a global
   optimum.

The baseline must be recorded before the first attempt. If it is not already
in `../RESULTS.tsv`, run:

```bash
python ../run_attempt.py --run-dir .. baseline
```

## What to preserve in reasoning

For each proposal, distinguish a parameterization-preserving compression from
a representational change (for example, token representation, positional
integration, deterministic/tied projections, attention organization, or
feed-forward mechanism). The source snapshots and Codex event log are research
artifacts, not scratch files.
"""


AGENT_PROMPT = """Read `PROGRAM.md` and operate as the autonomous researcher described there.

First confirm that the baseline has been recorded. Then perform logged
experiments until the run's configured attempt budget is exhausted. Do not ask
the operator to choose research ideas; make and test your own bounded,
well-documented choices. Follow every scope, anti-leakage, integrity, and
rollback requirement in `PROGRAM.md`. Your final response must state the run
directory and the number of attempts actually recorded, without claiming that
the observed frontier is a global optimum.
"""


LAUNCH_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

CODEX_MODEL=__CODEX_MODEL__
CODEX_REASONING_EFFORT=__CODEX_REASONING_EFFORT__
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$RUN_DIR/workspace"
mkdir -p "$RUN_DIR/logs"

# This starts a real Codex agent session. It can consume your configured Codex
# or API allowance; inspect RUN_CONFIG.json before launching.
codex exec --model "$CODEX_MODEL" \\
  -c "model_reasoning_effort=$CODEX_REASONING_EFFORT" \\
  --json --approve-for-me \\
  --cd "$WORKSPACE" --add-dir "$RUN_DIR" \\
  --output-last-message "$RUN_DIR/logs/codex-last-message.md" \\
  "$(cat "$RUN_DIR/AGENT_PROMPT.md")" \\
  | tee "$RUN_DIR/logs/codex-events.jsonl"
"""


def _launch_script(model: str, reasoning_effort: str) -> str:
    """Render a shell-safe launcher with the run's explicit model settings."""
    return LAUNCH_SCRIPT.replace(
        "__CODEX_MODEL__", shlex.quote(model)
    ).replace("__CODEX_REASONING_EFFORT__", shlex.quote(reasoning_effort))


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
        shutil.copy2(
            STARTING_MODEL / "checkpoints" / "best.pt",
            workspace / "checkpoints" / "best.pt",
        )
        (workspace / "submission.py").write_text(SUBMISSION_WRAPPER, encoding="utf-8")
        (workspace / "PROGRAM.md").write_text(PROGRAM, encoding="utf-8")
        (workspace / ".gitignore").write_text(
            "checkpoints/\nresults/\n__pycache__/\n*.py[cod]\n",
            encoding="utf-8",
        )

        shutil.copy2(ADDERBOARD / "verify.py", run_dir / "official_verify.py")
        shutil.copy2(RUNNER, run_runner)
        (run_dir / "attempts").mkdir()
        (run_dir / "state").mkdir()
        (run_dir / "logs").mkdir()
        (run_dir / "AGENT_PROMPT.md").write_text(AGENT_PROMPT, encoding="utf-8")
        (run_dir / "launch_codex.sh").write_text(
            _launch_script(args.agent_model, args.agent_reasoning_effort),
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
        choices=("minimal", "low", "medium", "high", "xhigh"),
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")
    if args.training_timeout < 1 or args.verification_timeout < 1:
        raise SystemExit("timeouts must be positive")
    run_dir = create_run(args)
    print(run_dir)
    print(
        f"Next: {args.python_bin} {run_dir / 'run_attempt.py'} --run-dir {run_dir} baseline"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
