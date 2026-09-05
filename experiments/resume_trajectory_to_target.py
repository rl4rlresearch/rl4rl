"""Resume one trajectory and cooperatively pause at an exact proposal boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.c0c3_factorial import orchestration
from experiments.c0c3_factorial.cli import _load_campaign
from experiments.c0c3_factorial.state import SearchController


def resume_to_target(campaign, run_id, target, *, repo_root, python_bin, codex_binary):
    spec, task, framework = _load_campaign(campaign)
    state = SearchController.load(campaign / "runs" / run_id, spec).state
    if state.condition == "C4":
        raise ValueError("C4 is excluded from this continuation")
    if state.active is not None:
        raise ValueError("Recover the active proposal before resuming")
    if not state.proposals_used < target < spec.budget.proposals:
        raise ValueError("Target must exceed current count and precede campaign end")
    original = orchestration.run_one_opportunity

    def bounded_opportunity(run_dir, **kwargs):
        current = SearchController.load(run_dir, spec).state
        if current.proposals_used >= target:
            raise RuntimeError("Refusing to begin a proposal beyond the target")
        result = original(run_dir, **kwargs)
        if int(result["proposals_cumulative"]) >= target:
            # Synchronous request before the controller can begin another call.
            orchestration.request_staged_trajectory_pause(
                campaign,
                spec=spec,
                run_id=run_id,
                reason=f"Operator-requested pause at {target} proposals",
            )
        return result

    orchestration.run_one_opportunity = bounded_opportunity
    try:
        return orchestration.run_staged_individual_trajectory(
            campaign,
            spec=spec,
            task=task,
            framework=framework,
            repo_root=repo_root,
            python_bin=python_bin,
            run_id=run_id,
            resume=True,
            codex_binary=codex_binary,
        )
    finally:
        orchestration.run_one_opportunity = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--codex-binary", required=True)
    args = parser.parse_args()
    result = resume_to_target(
        args.campaign.resolve(),
        args.run_id,
        args.target,
        repo_root=Path(__file__).resolve().parents[1],
        python_bin=args.python_bin,
        codex_binary=args.codex_binary,
    )
    print(json.dumps(result, indent=2))
