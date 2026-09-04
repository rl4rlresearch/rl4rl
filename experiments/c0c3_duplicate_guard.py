"""Condition-common duplicate enforcement for hash-pinned C0-C3 runtimes.

Older runners kept only qualified candidates in ``state.candidates``.  As a
result, byte-identical source that had already failed inside the evaluator was
evaluated again, and duplicates of qualified candidates were reported under a
generic ``invalid_candidate`` label.  This operational compatibility patch
restores the protocol's exact-content rule without editing a campaign's
hash-pinned runtime tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

DUPLICATE_MESSAGE = "proposal reproduced an evaluated candidate"


def _completed_evaluations(run_dir: Path) -> dict[str, int]:
    """Map evaluated candidate content hashes to their first opportunity."""

    events = run_dir / "events.jsonl"
    if not events.is_file():
        return {}
    evaluated: dict[str, int] = {}
    for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") != "proposal_completed":
            continue
        evaluation = record.get("evaluation")
        if not isinstance(evaluation, dict) or evaluation.get("evaluator_calls") != 1:
            continue
        artifact_path = record.get("artifact_path")
        opportunity = record.get("opportunity")
        if not isinstance(artifact_path, str) or not isinstance(opportunity, int):
            continue
        candidate_id = Path(artifact_path).name
        if candidate_id:
            evaluated.setdefault(candidate_id, opportunity)
    return evaluated


def install_duplicate_guard(
    runner_module: ModuleType,
    state_module: ModuleType,
    evaluator_module: ModuleType,
) -> None:
    """Install idempotent exact-content rejection around a frozen runner."""

    if getattr(runner_module, "_rl4rl_duplicate_guard_installed", False):
        return

    original_make_evaluator = runner_module.make_command_evaluator

    def make_guarded_evaluator(**kwargs: Any) -> Any:
        underlying = original_make_evaluator(**kwargs)

        class GuardedEvaluator:
            def __getattr__(self, name: str) -> Any:
                return getattr(underlying, name)

            def evaluate(
                self,
                *,
                candidate_snapshot: Path,
                opportunity_root: Path,
                timeout_seconds: int,
                run_seed: int | None = None,
            ) -> Any:
                run_dir = opportunity_root.parent.parent
                candidate_id = candidate_snapshot.name
                first_opportunity = _completed_evaluations(run_dir).get(candidate_id)
                if first_opportunity is None:
                    return underlying.evaluate(
                        candidate_snapshot=candidate_snapshot,
                        opportunity_root=opportunity_root,
                        timeout_seconds=timeout_seconds,
                        run_seed=run_seed,
                    )
                receipt = {
                    "schema_version": "1.0",
                    "event": "duplicate_candidate_rejected_before_evaluation",
                    "candidate_id": candidate_id,
                    "first_evaluated_opportunity": first_opportunity,
                }
                (opportunity_root / "duplicate-rejection.json").write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                evaluation = state_module.Evaluation(
                    valid=False,
                    fitness=None,
                    metrics={},
                    evaluator_seconds=0.0,
                    evaluator_calls=0,
                    failure_kind="duplicate",
                )
                return evaluator_module.EvaluationArtifacts(
                    evaluation=evaluation,
                    stdout_path=opportunity_root / "evaluation.stdout.log",
                    stderr_path=opportunity_root / "evaluation.stderr.log",
                    workspace_path=opportunity_root / "evaluation-workspace",
                )

        return GuardedEvaluator()

    original_complete = state_module.SearchController.complete

    def complete_with_duplicate_kind(self: Any, *args: Any, **kwargs: Any) -> Any:
        evaluation = kwargs.get("evaluation")
        if evaluation is not None:
            metrics = evaluation.metrics
            if (
                evaluation.failure_kind == "invalid_candidate"
                and isinstance(metrics, dict)
                and metrics.get("adapter_error") == DUPLICATE_MESSAGE
            ):
                kwargs["evaluation"] = state_module.Evaluation(
                    valid=False,
                    fitness=None,
                    metrics=metrics,
                    evaluator_seconds=evaluation.evaluator_seconds,
                    evaluator_calls=evaluation.evaluator_calls,
                    failure_kind="duplicate",
                )
        return original_complete(self, *args, **kwargs)

    runner_module.make_command_evaluator = make_guarded_evaluator
    state_module.SearchController.complete = complete_with_duplicate_kind
    runner_module._rl4rl_duplicate_guard_installed = True
