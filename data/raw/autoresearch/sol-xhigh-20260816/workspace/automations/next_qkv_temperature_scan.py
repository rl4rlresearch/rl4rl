#!/usr/bin/env python3
"""Calibrate temperature around the next one-example QKV union boundary."""
from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path
WORKSPACE=Path(__file__).resolve().parents[1]; RUN_DIR=WORKSPACE.parent; MODEL=WORKSPACE/"src"/"model.py"
TEMPS=(1.0125,1.0175,1.01,1.02,1.005,1.025)
def rows(p):
    with p.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h,delimiter="\t"))
def make(temp):
    s=MODEL.read_text(encoding="utf-8"); start=s.index("QKV_TIED_PAIRS = ("); end=s.index("\n)\n\nFF_IN_TIED_PAIRS",start)
    s=s[:end]+"\n    (608, 539),"+s[end:]
    old="(1.015 / math.sqrt(self.d_head))"
    if old not in s: raise RuntimeError("temperature marker not found")
    MODEL.write_text(s.replace(old,f"({temp!r} / math.sqrt(self.d_head))",1),encoding="utf-8")
def main():
    while True:
        st=json.loads((RUN_DIR/"STATE.json").read_text()); a=st["active_automation"]; u=int(a["micro_attempts_used"])
        if st["incumbent"]["parameters"]<a["parent_parameters"]: print("Qualified; equal-count scan exhausted."); return 0
        if u>=len(TEMPS) or u>=int(a["max_micro_trials"]): print("Temperature cap reached."); return 0
        t=TEMPS[u]; make(t)
        p=(f"Family: attention organization. Current retained frontier is 1,640 parameters at 99.000000%, exactly threshold. "
           f"There have been 19 prior macro-attempts and {u} prior micro-trials in this policy. The most recent accepted result is "
           f"attempt-0146/micro-0011 at 1,640/99.00%; the most recent failed result is attempt-0146/micro-0012, where QKV pair "
           f"608/539 scored 98.99% at 1,639 with temperature 1.015. Test the identical pair at absolute multiplier {t}. This is more "
           f"informative than the next scheduled temperature because it is the smallest remaining deviation around 1.015. Official "
           f"acceptance/rollback is unchanged; acceptance exhausts equal-count eligibility.")
        d=subprocess.run([sys.executable,str(RUN_DIR/"run_attempt.py"),"--run-dir",str(RUN_DIR),"automation-attempt","--description",f"Use temperature {t} with QKV union 608/539","--proposal",p],cwd=WORKSPACE,capture_output=True,text=True)
        print(d.stdout,end="");print(d.stderr,end="",file=sys.stderr)
        if d.returncode!=0 or rows(RUN_DIR/"AUTOMATION_RESULTS.tsv")[-1]["status"]=="error":return d.returncode
if __name__=="__main__":raise SystemExit(main())
