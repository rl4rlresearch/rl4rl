#!/usr/bin/env python3
"""Test remaining aligned operand-position row ties in distance order."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[1]
RUN_DIR = WORKSPACE.parent
MODEL = WORKSPACE / "src" / "model.py"
EXCLUDED = {(1, 2)}


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    while True:
        state = json.loads((RUN_DIR / "STATE.json").read_text())
        active = state["active_automation"]
        used = int(active["micro_attempts_used"])
        if used >= int(active["max_micro_trials"]):
            print("Reached declared micro-trial cap.", flush=True)
            return 0
        checkpoint = torch.load(
            RUN_DIR / "state" / "incumbent.pt", map_location="cpu", weights_only=False
        )["model_state"]
        positions = checkpoint["pos_emb.values"][checkpoint["pos_emb.value_indices"]]
        candidates = []
        for column in range(1, 10):
            left, right = 1 + 2 * column, 2 + 2 * column
            if (left, right) not in EXCLUDED:
                candidates.append((float((positions[left] - positions[right]).norm()), left, right))
        candidates.sort()
        if not candidates:
            print("No eligible operand-position pair remains.", flush=True)
            return 0
        distance, left, right = candidates[0]
        alternative = candidates[1][0] if len(candidates) > 1 else float("nan")
        source = MODEL.read_text(encoding="utf-8")
        marker = "                        (116, 11)))"
        replacement = (
            "                        (116, 11)) + tuple(\n"
            f"                            ({left} * d_model + feature, {right} * d_model + feature)\n"
            "                            for feature in range(d_model)))"
        )
        if marker not in source:
            raise RuntimeError("position tie insertion marker not found")
        MODEL.write_text(source.replace(marker, replacement, 1), encoding="utf-8")
        proposal = (
            f"Family: position representation. Current retained frontier is "
            f"{state['incumbent']['parameters']:,} parameters at 99.010000%, a +0.010000 "
            f"percentage-point margin over 99%. There have been 9 prior macro-attempts and "
            f"{used} prior micro-trials in this policy. The most recent accepted result is "
            f"attempt-0030 at 2,128 parameters/99.14%; the most recent failed result is "
            f"attempt-0131, whose position-1/2 row tie scored 15.91% at 1,656 parameters. "
            f"Tie operand-position rows {left} and {right} coordinate-wise; their checkpoint "
            f"L2 distance is {distance:.6f}, versus {alternative:.6f} for the next eligible "
            f"column. This is the least-distortion remaining test and is more informative than "
            f"the next row pair for that reason. The official runner alone accepts or rolls back; "
            f"the first discard or error closes this column-invariance path."
        )
        completed = subprocess.run(
            [sys.executable, str(RUN_DIR / "run_attempt.py"), "--run-dir", str(RUN_DIR),
             "automation-attempt", "--description",
             f"Tie operand-position rows {left} and {right} coordinate-wise",
             "--proposal", proposal], cwd=WORKSPACE, capture_output=True, text=True)
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            return completed.returncode
        result = rows(RUN_DIR / "AUTOMATION_RESULTS.tsv")[-1]
        if result["status"] in {"discard", "error"}:
            print(f"Recorded {result['status']} closes operand row-tie path.", flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
