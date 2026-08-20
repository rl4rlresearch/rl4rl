"""Run one bounded, runner-mediated monotonic feed-forward width sweep."""

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
WIDTHS = (16, 12, 8, 4)


def active_dir() -> Path:
    state = json.loads((RUN_DIR / "STATE.json").read_text())
    return RUN_DIR / "automations" / state["active_automation"]["attempt_id"]


def write_trigger(reason: str, last_micro_trial: str | None, detail: str) -> None:
    payload = {
        "reason": reason,
        "last_micro_trial_id": last_micro_trial,
        "detail": detail,
    }
    (active_dir() / "AUTOMATION_TRIGGER.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def set_width(width: int) -> None:
    source = TRAIN.read_text()
    updated, replacements = re.subn(
        r"'ff_dim': \d+", f"'ff_dim': {width}", source, count=1
    )
    if replacements != 1:
        raise RuntimeError("Could not locate exactly one feed-forward width setting.")
    TRAIN.write_text(updated)


def latest_row() -> dict[str, str]:
    with (RUN_DIR / "AUTOMATION_RESULTS.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError("Runner returned without recording a micro-trial.")
    return rows[-1]


def main() -> int:
    for width in WIDTHS:
        try:
            set_width(width)
        except Exception as exc:
            write_trigger("runner-failure", None, f"candidate edit failed: {exc}")
            return 1

        proposal = (
            "Family: feed-forward width. Automation decision: test the next "
            f"descending eligible bottleneck width, ff={width}, after all larger "
            "listed widths have either been retained or ruled out by the runner. "
            "This is parameterization-preserving scalar compression; the runner "
            "is the sole training, verification, and retention authority."
        )
        result = subprocess.run(
            [
                sys.executable,
                str(RUN_DIR / "run_attempt.py"),
                "--run-dir",
                str(RUN_DIR),
                "automation-attempt",
                "--description",
                f"Automation micro-trial: reduce feed-forward width to {width}.",
                "--proposal",
                proposal,
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
            # The runner has already written its required scored-discard/error trigger.
            return 0

    # Reaching the declared cap is a runner trigger; do not overwrite it here.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
