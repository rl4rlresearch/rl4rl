from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from experiments import c0c3_duplicate_guard
from experiments.c0c3_factorial.evaluator import EvaluationArtifacts
from experiments.c0c3_factorial.runner import _evaluated_candidate_ids
from experiments.c0c3_factorial.state import Evaluation


class _UnderlyingEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, **_kwargs: object) -> EvaluationArtifacts:
        self.calls += 1
        return EvaluationArtifacts(
            evaluation=Evaluation(
                valid=False,
                fitness=None,
                metrics={"accuracy": 0.5},
                evaluator_seconds=1.0,
                evaluator_calls=1,
                failure_kind="nonqualification",
            ),
            stdout_path=Path("stdout"),
            stderr_path=Path("stderr"),
            workspace_path=Path("workspace"),
        )


class _FakeController:
    def complete(self, **kwargs: object) -> Evaluation:
        return kwargs["evaluation"]  # type: ignore[return-value]


def _modules(underlying: _UnderlyingEvaluator) -> tuple[object, object, object]:
    runner = SimpleNamespace(make_command_evaluator=lambda **_kwargs: underlying)
    state = SimpleNamespace(SearchController=_FakeController, Evaluation=Evaluation)
    evaluator = SimpleNamespace(EvaluationArtifacts=EvaluationArtifacts)
    return runner, state, evaluator


def test_repeated_failed_source_is_rejected_before_another_evaluator_call(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    opportunity_root = run_dir / "opportunities" / "0002"
    candidate = run_dir / "candidates" / ("a" * 64)
    opportunity_root.mkdir(parents=True)
    candidate.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "proposal_completed",
                "opportunity": 1,
                "artifact_path": f"candidates/{candidate.name}",
                "evaluation": {
                    "valid": False,
                    "evaluator_calls": 1,
                    "failure_kind": "execution",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    underlying = _UnderlyingEvaluator()
    runner, state, evaluator = _modules(underlying)
    c0c3_duplicate_guard.install_duplicate_guard(runner, state, evaluator)

    guarded = runner.make_command_evaluator()
    result = guarded.evaluate(
        candidate_snapshot=candidate,
        opportunity_root=opportunity_root,
        timeout_seconds=10,
        run_seed=7,
    )

    assert underlying.calls == 0
    assert result.evaluation.failure_kind == "duplicate"
    assert result.evaluation.evaluator_calls == 0
    receipt = json.loads((opportunity_root / "duplicate-rejection.json").read_text())
    assert receipt["first_evaluated_opportunity"] == 1


def test_novel_source_still_reaches_the_evaluator(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    opportunity_root = run_dir / "opportunities" / "0001"
    candidate = run_dir / "candidates" / ("b" * 64)
    opportunity_root.mkdir(parents=True)
    candidate.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    underlying = _UnderlyingEvaluator()
    runner, state, evaluator = _modules(underlying)
    c0c3_duplicate_guard.install_duplicate_guard(runner, state, evaluator)

    result = runner.make_command_evaluator().evaluate(
        candidate_snapshot=candidate,
        opportunity_root=opportunity_root,
        timeout_seconds=10,
        run_seed=7,
    )

    assert underlying.calls == 1
    assert result.evaluation.failure_kind == "nonqualification"


def test_existing_population_duplicate_is_reported_as_duplicate() -> None:
    underlying = _UnderlyingEvaluator()
    runner, state, evaluator = _modules(underlying)
    c0c3_duplicate_guard.install_duplicate_guard(runner, state, evaluator)
    original = Evaluation(
        valid=False,
        fitness=None,
        metrics={
            "adapter_error": c0c3_duplicate_guard.DUPLICATE_MESSAGE,
            "codex_returncode": 0,
        },
        evaluator_seconds=0.0,
        evaluator_calls=0,
        failure_kind="invalid_candidate",
    )

    repaired = state.SearchController().complete(evaluation=original)

    assert repaired.failure_kind == "duplicate"
    assert repaired.metrics == original.metrics


def test_runner_duplicate_index_includes_invalid_evaluations(tmp_path: Path) -> None:
    candidate_id = "c" * 64
    (tmp_path / "events.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "event": "proposal_completed",
                    "artifact_path": f"candidates/{candidate_id}",
                    "evaluation": {
                        "valid": False,
                        "evaluator_calls": 1,
                        "failure_kind": "execution",
                    },
                },
                {
                    "event": "proposal_completed",
                    "artifact_path": f"candidates/{'d' * 64}",
                    "evaluation": {
                        "valid": False,
                        "evaluator_calls": 0,
                        "failure_kind": "source_preflight",
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert _evaluated_candidate_ids(tmp_path) == {candidate_id}
