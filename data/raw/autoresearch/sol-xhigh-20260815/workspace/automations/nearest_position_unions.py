#!/usr/bin/env python3
"""Run one bounded macro-search over nearest learned position-value unions."""

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
BASE_PRIOR_FAMILY_ATTEMPTS = 18


def state() -> dict:
    return json.loads((RUN_DIR / "STATE.json").read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_family_evidence() -> tuple[str, str]:
    accepted = "attempt-0074 at 2,105 parameters and 99.05%"
    failed = "attempt-0034 (shared normalization biases)"
    for row in rows(RUN_DIR / "AUTOMATION_RESULTS.tsv"):
        if row["family"] != "parameter tying":
            continue
        label = f"{row['macro_attempt_id']}/{row['micro_attempt_id']}"
        if row["status"] == "keep":
            accepted = f"{label} at {row['parameters']} parameters and {row['accuracy']}"
        elif row["status"] in {"discard", "error"}:
            failed = f"{label} ({row['status']}, {row['accuracy'] or 'no score'})"
    return accepted, failed


def candidate_pairs(excluded: set[tuple[int, int]]) -> list[tuple[float, int, int, float, float]]:
    checkpoint = torch.load(
        RUN_DIR / "state" / "incumbent.pt", map_location="cpu", weights_only=False
    )["model_state"]
    values = checkpoint["pos_emb.values"]
    mapping = checkpoint["pos_emb.value_indices"].flatten()
    sorted_values, sorted_groups = torch.sort(values)
    candidates: list[tuple[float, int, int, float, float]] = []
    for index in range(len(sorted_values) - 1):
        left_group = int(sorted_groups[index])
        right_group = int(sorted_groups[index + 1])
        left_coord = int((mapping == left_group).nonzero()[0])
        right_coord = int((mapping == right_group).nonzero()[0])
        key = tuple(sorted((left_coord, right_coord)))
        if key in excluded:
            continue
        left = float(values[left_group])
        right = float(values[right_group])
        candidates.append((abs(right - left), left_coord, right_coord, left, right))
    candidates.sort()
    return candidates


def add_pair(left: int, right: int) -> None:
    source = MODEL.read_text(encoding="utf-8")
    boundary = source.index("\n\n        self.blocks")
    closing = source.rfind(")))", 0, boundary)
    if closing < 0:
        raise RuntimeError("could not locate tied_pairs terminator")
    # The first of the three closing parentheses terminates the existing pair;
    # retain it while inserting the comma and the new pair.
    replacement = f"),\n                        ({left}, {right})))"
    MODEL.write_text(source[:closing] + replacement + source[closing + 3 :], encoding="utf-8")


def main() -> int:
    excluded: set[tuple[int, int]] = set()
    while True:
        current = state()
        automation = current["active_automation"]
        used = int(automation["micro_attempts_used"])
        cap = int(automation["max_micro_trials"])
        if used >= cap:
            print(f"Reached declared micro-trial cap ({cap}).", flush=True)
            return 0

        candidates = candidate_pairs(excluded)
        if not candidates:
            print("No eligible positional group pair remains.", flush=True)
            return 0
        difference, left, right, left_value, right_value = candidates[0]
        alternative = candidates[1][0] if len(candidates) > 1 else float("nan")
        accepted, failed = latest_family_evidence()
        parameters = int(current["incumbent"]["parameters"])
        accuracy_text = automation.get("best_accuracy") or "99.050000%"
        accuracy = float(str(accuracy_text).rstrip("%"))
        prior = BASE_PRIOR_FAMILY_ATTEMPTS + used

        add_pair(left, right)
        description = f"Union closest position groups {left} and {right}"
        proposal = (
            f"Family: parameter tying. Current retained frontier is {parameters:,} parameters "
            f"at {accuracy:.6f}%, a +{accuracy - 99:.6f} percentage-point margin over 99%. "
            f"There have been {prior} prior attempts in this family. The most recent accepted "
            f"result is {accepted}; the most recent failed result is {failed}. This is a "
            f"parameterization-preserving compression: union current position groups represented "
            f"by coordinates {left} and {right}, whose learned values are {left_value:.10f} and "
            f"{right_value:.10f} (absolute difference {difference:.3e}), using a multiplicity-weighted "
            f"mean. Hypothesis: this preserves qualification and yields {parameters - 1:,} parameters. "
            f"It is more informative than the nearest untested alternative (difference "
            f"{alternative:.3e}) because it is the minimum-distortion eligible union; exact pairs "
            f"already rejected under the unchanged grouping are excluded. Acceptance and rollback "
            f"remain solely governed by the official runner."
        )
        print(f"micro {used + 1}/{cap}: {left},{right} diff={difference:.3e}", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(RUN_DIR / "run_attempt.py"),
                "--run-dir",
                str(RUN_DIR),
                "automation-attempt",
                "--description",
                description,
                "--proposal",
                proposal,
            ],
            cwd=WORKSPACE,
        )
        if completed.returncode != 0:
            print(f"Official runner exited {completed.returncode}; stopping.", flush=True)
            return completed.returncode

        result = rows(RUN_DIR / "AUTOMATION_RESULTS.tsv")[-1]
        if result["status"] == "error":
            print("Micro-trial recorded an error; stopping for log inspection.", flush=True)
            return 0
        if result["status"] == "discard":
            excluded.add(tuple(sorted((left, right))))


if __name__ == "__main__":
    raise SystemExit(main())
