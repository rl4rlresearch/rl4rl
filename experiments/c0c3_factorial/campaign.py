"""Campaign calibration and construction for local or Modal execution."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from .artifacts import (
    prepare_seed_workspace,
    snapshot_candidate,
    tree_hash,
)
from .evaluator import CommandEvaluator
from .spec import (
    Condition,
    FactorialSpec,
    FrameworkSpec,
    TaskSpec,
    make_assignments,
    sha256_json,
)
from .state import Candidate, SearchController


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def calibrate_task(
    output_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    repo_root: Path,
    python_bin: str,
) -> Path:
    """Evaluate the frozen seed once on the target execution environment."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    support = output / "task-support"
    prepare_seed_workspace(task, support, repo_root=repo_root)
    candidate_id, snapshot = snapshot_candidate(
        support, output / "candidates", task.editable_paths
    )
    evaluator = CommandEvaluator(
        task=task,
        support_source=support,
        repo_root=repo_root,
        python_bin=python_bin,
    )
    artifacts = evaluator.evaluate(
        candidate_snapshot=snapshot,
        opportunity_root=output / "evaluation",
        timeout_seconds=spec.budget.evaluator_timeout_seconds,
    )
    if not artifacts.evaluation.valid:
        raise RuntimeError(
            f"seed calibration failed: {artifacts.evaluation.failure_kind}"
        )
    calibration = {
        "schema_version": "1.0",
        "task_id": task.task_id,
        "candidate_id": candidate_id,
        "support_tree_sha256": tree_hash(support),
        "fitness": artifacts.evaluation.fitness,
        "metrics": artifacts.evaluation.metrics,
        "evaluator_seconds": artifacts.evaluation.evaluator_seconds,
        "protocol_hash": spec.protocol_hash,
        "calibration_kind": "executed_on_target_backend",
    }
    path = output / "baseline.json"
    _write_json(path, calibration)
    return path


def _load_calibration(
    path: Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    candidate_id: str,
    support_hash: str,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "1.0",
        "task_id": task.task_id,
        "candidate_id": candidate_id,
        "support_tree_sha256": support_hash,
        "protocol_hash": spec.protocol_hash,
        "calibration_kind": "executed_on_target_backend",
    }
    mismatch = {
        key: (value, payload.get(key))
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"baseline calibration does not match campaign: {mismatch}")
    if not isinstance(payload.get("fitness"), (int, float)):
        raise ValueError("baseline calibration lacks numeric fitness")
    if not isinstance(payload.get("metrics"), dict):
        raise ValueError("baseline calibration lacks metrics")
    return payload


def create_campaign(
    output_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    calibration_path: str | Path,
    repo_root: Path,
    include_no_search: bool = True,
) -> Path:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    common_support = output / "task-support"
    prepare_seed_workspace(task, common_support, repo_root=repo_root)
    candidate_id, common_snapshot = snapshot_candidate(
        common_support, output / "seed-candidate", task.editable_paths
    )
    calibration = _load_calibration(
        Path(calibration_path),
        spec=spec,
        task=task,
        candidate_id=candidate_id,
        support_hash=tree_hash(common_support),
    )
    assignments = list(
        make_assignments(
            spec,
            task_id=task.task_id,
            framework_id=framework.framework_id.value,
        )
    )
    schedule: list[dict[str, object]] = []
    for assignment in assignments:
        schedule.append(asdict(assignment) | {"condition": assignment.condition.value})
    if include_no_search:
        for block in range(1, spec.blocks + 1):
            paired = next(row for row in assignments if row.block == block)
            schedule.append(
                {
                    "block": block,
                    "order": 5,
                    "condition": "N0",
                    "run_seed": paired.run_seed,
                    "run_id": (
                        f"{spec.study_id}-{task.task_id}-"
                        f"{framework.framework_id.value}-b{block:02d}-n0"
                    ),
                }
            )
    schedule.sort(key=lambda row: (int(row["block"]), int(row["order"])))
    for assignment in schedule:
        run_id = str(assignment["run_id"])
        condition_label = str(assignment["condition"])
        no_search = condition_label == "N0"
        run_dir = output / "runs" / run_id
        seed = Candidate(
            candidate_id=candidate_id,
            parent_ids=[],
            fitness=float(calibration["fitness"]),
            metrics=dict(calibration["metrics"]),
            artifact_path=f"candidates/{candidate_id}",
            hypothesis="frozen seed baseline",
            intended_edit="none",
            created_opportunity=0,
            retained_order=0,
        )
        SearchController.create(
            run_dir,
            spec,
            run_id=run_id,
            condition=Condition.C0 if no_search else Condition(condition_label),
            seed_candidate=seed,
            no_search=no_search,
        )
        shutil.copytree(common_support, run_dir / "task-support")
        destination = run_dir / "candidates" / candidate_id
        destination.parent.mkdir()
        shutil.copytree(common_snapshot, destination)
        _write_json(
            run_dir / "manifest.json",
            {
                "schema_version": "1.0",
                "assignment": assignment,
                "protocol_hash": spec.protocol_hash,
                "task_hash": sha256_json(asdict(task)),
                "framework_hash": sha256_json(asdict(framework)),
                "baseline": calibration,
                "repo_revision": _repo_revision(repo_root),
            },
        )
    inputs = output / "inputs"
    inputs.mkdir()
    _write_json(inputs / "protocol.json", asdict(spec))
    _write_json(inputs / "task.json", asdict(task))
    _write_json(inputs / "framework.json", asdict(framework))
    _write_json(output / "schedule.json", schedule)
    _write_json(
        output / "campaign.json",
        {
            "schema_version": "1.0",
            "study_id": spec.study_id,
            "task_id": task.task_id,
            "framework_id": framework.framework_id.value,
            "protocol_hash": spec.protocol_hash,
            "task_hash": sha256_json(asdict(task)),
            "framework_hash": sha256_json(asdict(framework)),
            "seed_candidate_id": candidate_id,
            "include_no_search": include_no_search,
            "run_count": len(schedule),
        },
    )
    shutil.rmtree(output / "seed-candidate")
    shutil.rmtree(common_support)
    return output


def _repo_revision(repo_root: Path) -> str:
    head = repo_root / ".git/HEAD"
    if not head.is_file():
        return "unavailable"
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref: "):
        reference = repo_root / ".git" / content[5:]
        if reference.is_file():
            return reference.read_text(encoding="utf-8").strip()
    if len(content) == 40:
        return content
    return hashlib.sha256(content.encode()).hexdigest()
