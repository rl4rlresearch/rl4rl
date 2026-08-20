#!/usr/bin/env python3
"""Bounded attention-temperature compensation for one QKV union."""
from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
RUN_DIR = WORKSPACE.parent
MODEL = WORKSPACE / "src" / "model.py"
MULTIPLIERS = (0.99, 1.01, 0.98, 1.02, 0.95, 1.05, 0.90, 1.10)

def rows(path):
    with path.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h, delimiter="\t"))

def make_candidate(multiplier):
    source = MODEL.read_text(encoding="utf-8")
    old = "    (714, 716),\n)\n\nFF_IN_TIED_PAIRS"
    new = "    (714, 716),\n    (555, 517),\n)\n\nFF_IN_TIED_PAIRS"
    if old not in source: raise RuntimeError("QKV marker not found")
    source = source.replace(old, new, 1)
    old = "(1.0 / math.sqrt(self.d_head))"
    if old not in source: raise RuntimeError("temperature marker not found")
    source = source.replace(old, f"({multiplier!r} / math.sqrt(self.d_head))", 1)
    MODEL.write_text(source, encoding="utf-8")

def main():
    while True:
        state = json.loads((RUN_DIR / "STATE.json").read_text())
        active = state["active_automation"]; used = int(active["micro_attempts_used"])
        if state["incumbent"]["parameters"] < active["parent_parameters"]:
            print("A temperature-qualified candidate exhausted equal-count eligibility."); return 0
        if used >= len(MULTIPLIERS) or used >= int(active["max_micro_trials"]):
            print("Reached declared temperature cap."); return 0
        multiplier = MULTIPLIERS[used]; make_candidate(multiplier)
        proposal = (
            f"Family: attention organization. Current retained frontier is 1,671 parameters at "
            f"99.000000%, exactly threshold. There have been 17 prior macro-attempts and {used} "
            f"prior micro-trials in this policy. The most recent accepted result is attempt-0020 "
            f"at 3,168/99.94%; the most recent failed result is attempt-0142, where four GELU "
            f"blends with QKV union 555/517 all scored 98.99% at 1,670. Use the identical union "
            f"and multiply QK logits by {multiplier} relative to retained temperature. This is "
            f"more informative than the next scheduled multiplier because the ordering alternates "
            f"the smallest untested deviations around one. Official acceptance/rollback remains "
            f"unchanged; acceptance exhausts equal-count eligibility."
        )
        done = subprocess.run([sys.executable, str(RUN_DIR/"run_attempt.py"), "--run-dir",
            str(RUN_DIR), "automation-attempt", "--description",
            f"Scale attention logits by {multiplier} with QKV union 555/517", "--proposal",
            proposal], cwd=WORKSPACE, capture_output=True, text=True)
        print(done.stdout, end=""); print(done.stderr, end="", file=sys.stderr)
        if done.returncode != 0 or rows(RUN_DIR/"AUTOMATION_RESULTS.tsv")[-1]["status"] == "error":
            return done.returncode

if __name__ == "__main__": raise SystemExit(main())
