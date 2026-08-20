#!/usr/bin/env python3
"""Bounded monotone blend from exact GELU to tanh-approximated GELU."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
RUN_DIR = WORKSPACE.parent
MODEL = WORKSPACE / "src" / "model.py"
ALPHAS = (0.5, 0.25, 0.125, 0.0)


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def make_candidate(alpha: float) -> None:
    source = MODEL.read_text(encoding="utf-8")
    pair_marker = "    (108, 165),\n)\n\nLN_F_TIED_PAIRS"
    pair_replacement = "    (108, 165),\n    (82, 85),\n)\n\nLN_F_TIED_PAIRS"
    if pair_marker not in source:
        raise RuntimeError("FF output-pair marker not found")
    source = source.replace(pair_marker, pair_replacement, 1)
    class_marker = "\n\nclass TiedScaleLayerNorm(nn.Module):"
    blend = f'''\n\nclass BlendedGELU(nn.Module):
    def forward(self, x):
        exact = F.gelu(x, approximate="none")
        approximate = F.gelu(x, approximate="tanh")
        return exact + {alpha!r} * (approximate - exact)
'''
    if class_marker not in source:
        raise RuntimeError("activation class marker not found")
    source = source.replace(class_marker, blend + class_marker, 1)
    source = source.replace("            nn.GELU(),", "            BlendedGELU(),", 1)
    MODEL.write_text(source, encoding="utf-8")


def main() -> int:
    while True:
        state = json.loads((RUN_DIR / "STATE.json").read_text())
        active = state["active_automation"]
        used = int(active["micro_attempts_used"])
        if state["incumbent"]["parameters"] < active["parent_parameters"]:
            print("An equal-count blend qualified; no eligible candidate remains.", flush=True)
            return 0
        if used >= len(ALPHAS) or used >= int(active["max_micro_trials"]):
            print("Reached declared interpolation boundary.", flush=True)
            return 0
        alpha = ALPHAS[used]
        make_candidate(alpha)
        proposal = (
            f"Family: feed-forward width. Current retained frontier is 1,672 parameters at "
            f"99.010000%, a +0.010000 percentage-point margin over 99%. There have been 20 "
            f"prior macro-attempts and {used} prior micro-trials in this interpolation policy. "
            f"The most recent accepted result is attempt-0022 at 3,040 parameters/99.85%; the "
            f"most recent failed result is attempt-0138, where alpha=1 tanh-GELU plus output "
            f"union 82/85 scored 98.99% at 1,671 parameters. Use activation exact_GELU + "
            f"{alpha}*(tanh_GELU-exact_GELU) with the identical union, retaining width 16. "
            f"This is more informative than the next smaller blend because it is the largest "
            f"remaining perturbation inside the measured alpha=0 incumbent/alpha=1 failure "
            f"bracket. Official acceptance and rollback are unchanged; a failure makes the "
            f"next smaller alpha eligible."
        )
        completed = subprocess.run(
            [sys.executable, str(RUN_DIR / "run_attempt.py"), "--run-dir", str(RUN_DIR),
             "automation-attempt", "--description",
             f"Blend exact and tanh GELU at alpha {alpha} with one output union",
             "--proposal", proposal], cwd=WORKSPACE, capture_output=True, text=True)
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            return completed.returncode
        result = rows(RUN_DIR / "AUTOMATION_RESULTS.tsv")[-1]
        if result["status"] == "error":
            print("Reproducible error closes interpolation.", flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
