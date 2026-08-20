#!/usr/bin/env python3
"""Bounded activation-blend compensation scan for one QKV scalar union."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
RUN_DIR = WORKSPACE.parent
MODEL = WORKSPACE / "src" / "model.py"
ALPHAS = (0.25, 0.75, 0.125, 0.875)


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def candidate(alpha: float) -> None:
    source = MODEL.read_text(encoding="utf-8")
    qkv_end = "    (714, 716),\n)\n\nFF_IN_TIED_PAIRS"
    replacement = "    (714, 716),\n    (555, 517),\n)\n\nFF_IN_TIED_PAIRS"
    if qkv_end not in source:
        raise RuntimeError("QKV insertion marker not found")
    source = source.replace(qkv_end, replacement, 1)
    old = "return exact + 0.5 * (approximate - exact)"
    if old not in source:
        raise RuntimeError("retained blend marker not found")
    source = source.replace(old, f"return exact + {alpha!r} * (approximate - exact)", 1)
    MODEL.write_text(source, encoding="utf-8")


def main() -> int:
    while True:
        state = json.loads((RUN_DIR / "STATE.json").read_text())
        active = state["active_automation"]
        used = int(active["micro_attempts_used"])
        if state["incumbent"]["parameters"] < active["parent_parameters"]:
            print("A 1,670-parameter blend qualified; no equal-count candidate remains eligible.")
            return 0
        if used >= len(ALPHAS) or used >= int(active["max_micro_trials"]):
            print("Reached declared four-point blend boundary.")
            return 0
        alpha = ALPHAS[used]
        candidate(alpha)
        proposal = (
            f"Family: feed-forward width. Current retained frontier is 1,671 parameters at "
            f"99.000000%, exactly threshold. There have been 21 prior macro-attempts and "
            f"{used} prior micro-trials in this compensation policy. The most recent accepted "
            f"result is attempt-0139/micro-0001 at 1,671/99.00%; the most recent failed result "
            f"is attempt-0141/micro-0001, QKV pair 555/517 at 1,670/98.99% with alpha=0.5. "
            f"Use the identical QKV union with blended-GELU alpha={alpha}. This is more "
            f"informative than the next scheduled alpha because it is the next widest untested "
            f"point around the alpha=0 and alpha=0.5 failures. Official acceptance/rollback is "
            f"unchanged; acceptance exhausts equal-count eligibility."
        )
        completed = subprocess.run(
            [sys.executable, str(RUN_DIR / "run_attempt.py"), "--run-dir", str(RUN_DIR),
             "automation-attempt", "--description",
             f"Use GELU blend alpha {alpha} with QKV union 555/517",
             "--proposal", proposal], cwd=WORKSPACE, capture_output=True, text=True)
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            return completed.returncode
        if rows(RUN_DIR / "AUTOMATION_RESULTS.tsv")[-1]["status"] == "error":
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
