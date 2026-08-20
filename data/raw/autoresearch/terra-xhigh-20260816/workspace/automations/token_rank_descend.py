"""Run the remaining bounded, runner-mediated tied-token-rank sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parents[1]
RUN_DIR = WORKSPACE.parent
TRAIN = WORKSPACE / "src" / "train.py"
RANKS = (3, 2, 1)


def active_dir() -> Path:
    state = json.loads((RUN_DIR / "STATE.json").read_text())
    return RUN_DIR / "automations" / state["active_automation"]["attempt_id"]


def write_trigger(reason: str, last_micro_trial: str | None, detail: str) -> None:
    (active_dir() / "AUTOMATION_TRIGGER.json").write_text(
        json.dumps(
            {
                "reason": reason,
                "last_micro_trial_id": last_micro_trial,
                "detail": detail,
            },
            indent=2,
        )
        + "\n"
    )


def set_rank(rank: int) -> None:
    source = TRAIN.read_text()
    updated, replacements = re.subn(
        r"'token_rank': \d+", f"'token_rank': {rank}", source, count=1
    )
    if replacements != 1:
        raise RuntimeError("Could not locate exactly one token-rank setting.")
    TRAIN.write_text(updated)


def latest_row() -> dict[str, str]:
    with (RUN_DIR / "AUTOMATION_RESULTS.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError("Runner returned without recording a micro-trial.")
    return rows[-1]


def main() -> int:
    for rank in RANKS:
        try:
            set_rank(rank)
        except Exception as exc:
            write_trigger("runner-failure", None, f"candidate edit failed: {exc}")
            return 1
        result = subprocess.run(
            [
                sys.executable,
                str(RUN_DIR / "run_attempt.py"),
                "--run-dir",
                str(RUN_DIR),
                "automation-attempt",
                "--description",
                f"Automation micro-trial: reduce learned tied token rank to {rank}.",
                "--proposal",
                "Family: token representation. Automation decision: test the next "
                f"descending eligible tied-token code rank, r={rank}, after all "
                "larger listed ranks have been retained or ruled out by the runner. "
                "This is token-subspace representational compression; the runner "
                "is the sole training, verification, and retention authority.",
            ],
            cwd=WORKSPACE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            write_trigger(
                "runner-failure",
                None,
                f"runner exited {result.returncode}: {result.stderr[-500:]}",
            )
            return result.returncode
        try:
            row = latest_row()
        except Exception as exc:
            write_trigger("runner-failure", None, f"result read failed: {exc}")
            return 1
        if row["status"] != "keep":
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
