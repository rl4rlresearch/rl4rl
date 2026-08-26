"""Multi-arm semantic intervention campaigns built on the v3 controller.

The original C0-C3 protocols remain intact.  This module adds a separate
trajectory-level design in which many research-process prompts branch from one
literal shared prefix per replicate. Every arm uses the same configured search
architecture, evaluator, evidence renderer, five-opportunity conversation
phases, budget, and scheduling machinery; only the registered semantic
direction differs at phase boundaries.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import shutil
import tomllib
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    prepare_seed_workspace,
    scientific_runtime_hash,
    snapshot_candidate,
    tree_hash,
)
from .campaign import _load_calibration, _repo_revision
from .frameworks import preload_framework_runtime
from .native_openevolve import is_native_openevolve, mirror_native_prefix_state
from .runner import recover_active_opportunity, run_one_opportunity
from .spec import (
    Condition,
    FactorialSpec,
    FrameworkSpec,
    TaskSpec,
    framework_hash_payload,
    sha256_json,
    task_hash_payload,
)
from .state import Candidate, SearchController, append_jsonl, atomic_json, utc_now
from .v3 import (
    V3_PROMPT_BUNDLE,
    _file_manifest,
    _json_hash,
    initialize_runtime_options,
    load_runtime_options,
    update_runtime_options,
    validate_prompt_bundle,
)

SEMANTIC_PROTOCOL_VERSION = "4.0"
SEMANTIC_MANIFEST = Path("semantic-interventions.json")
SEMANTIC_PREFIX = Path("semantic-prefix.json")
SEMANTIC_CONTROL = Path("semantic-control.json")
SEMANTIC_RUN_CONTROL = Path("semantic-run-control.json")
SEMANTIC_WAVES = Path("semantic-waves.jsonl")


@dataclass(frozen=True)
class Intervention:
    intervention_id: str
    label: str
    family: str
    prompt_path: str
    components: tuple[str, ...]


@dataclass(frozen=True)
class InterventionPlan:
    schema_version: str
    replicates: int
    shared_prefix_opportunities: int
    session_span_opportunities: int
    max_parallel_agent_calls: int
    task_evaluator_capacity: int
    interventions: tuple[Intervention, ...]

    @property
    def intervention_ids(self) -> tuple[str, ...]:
        return tuple(item.intervention_id for item in self.interventions)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_intervention_plan(path: str | Path) -> InterventionPlan:
    source = Path(path).resolve()
    payload = tomllib.loads(source.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "replicates",
        "shared_prefix_opportunities",
        "session_span_opportunities",
        "max_parallel_agent_calls",
        "task_evaluator_capacity",
        "interventions",
    }
    if set(payload) != expected:
        raise ValueError(
            "semantic intervention plan keys differ: "
            f"missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )
    interventions = []
    for row in payload["interventions"]:
        if set(row) != {"id", "label", "family", "prompt_path", "components"}:
            raise ValueError("semantic intervention row has invalid keys")
        interventions.append(
            Intervention(
                intervention_id=str(row["id"]),
                label=str(row["label"]),
                family=str(row["family"]),
                prompt_path=str(row["prompt_path"]),
                components=tuple(str(value) for value in row["components"]),
            )
        )
    plan = InterventionPlan(
        schema_version=str(payload["schema_version"]),
        replicates=int(payload["replicates"]),
        shared_prefix_opportunities=int(payload["shared_prefix_opportunities"]),
        session_span_opportunities=int(payload["session_span_opportunities"]),
        max_parallel_agent_calls=int(payload["max_parallel_agent_calls"]),
        task_evaluator_capacity=int(payload["task_evaluator_capacity"]),
        interventions=tuple(interventions),
    )
    if plan.schema_version != SEMANTIC_PROTOCOL_VERSION:
        raise ValueError("unsupported semantic intervention plan version")
    for name in (
        "replicates",
        "shared_prefix_opportunities",
        "session_span_opportunities",
        "task_evaluator_capacity",
    ):
        if getattr(plan, name) < 1:
            raise ValueError(f"{name} must be positive")
    if plan.max_parallel_agent_calls < 0:
        raise ValueError("max_parallel_agent_calls must be nonnegative")
    if len(plan.interventions) < 2:
        raise ValueError("semantic campaign requires at least two arms")
    if len(set(plan.intervention_ids)) != len(plan.interventions):
        raise ValueError("semantic intervention IDs must be unique")
    if "passive_control" not in plan.intervention_ids:
        raise ValueError("semantic plan requires passive_control for prefix ownership")
    for intervention in plan.interventions:
        path_value = Path(intervention.prompt_path)
        if path_value.is_absolute() or ".." in path_value.parts:
            raise ValueError("intervention prompt paths must be safe relative paths")
    return plan


def _derived_seed(study_seed: int, replicate: int) -> int:
    value = f"semantic-v4\0{study_seed}\0{replicate}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _schedule(
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    plan: InterventionPlan,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for replicate in range(1, plan.replicates + 1):
        order = list(plan.interventions)
        random.Random(_derived_seed(spec.study_seed, replicate)).shuffle(order)
        run_seed = _derived_seed(spec.study_seed, replicate)
        for position, intervention in enumerate(order, start=1):
            run_id = (
                f"{spec.study_id}-{task.task_id}-{framework.framework_key}-"
                f"r{replicate:02d}-{intervention.intervention_id}"
            )
            rows.append(
                {
                    "replicate": replicate,
                    "block": replicate,
                    "order": position,
                    "condition": intervention.intervention_id,
                    "condition_label": intervention.label,
                    "condition_family": intervention.family,
                    "components": list(intervention.components),
                    "controller_condition": Condition.C1.value,
                    "run_seed": run_seed,
                    "run_id": run_id,
                }
            )
    return sorted(rows, key=lambda row: (int(row["replicate"]), int(row["order"])))


def _snapshot_semantic_prompts(
    campaign: Path,
    *,
    spec: FactorialSpec,
    framework: FrameworkSpec,
    plan: InterventionPlan,
    repo_root: Path,
) -> dict[str, Any]:
    destination = campaign / V3_PROMPT_BUNDLE
    if destination.exists():
        raise FileExistsError("semantic campaign prompt bundle already exists")
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    shutil.copytree(repo_root / "experiments/c0c3_factorial/templates", temporary)
    prompt_source = (
        repo_root / "experiments/c0c3_factorial/templates/semantic_interventions_v4"
    )
    copied_root = temporary / "semantic-interventions"
    copied_root.mkdir()
    interventions = []
    for item in plan.interventions:
        source = prompt_source / item.prompt_path
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"semantic prompt is missing: {source}")
        destination_prompt = copied_root / item.prompt_path
        destination_prompt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_prompt)
        interventions.append(
            {
                "id": item.intervention_id,
                "label": item.label,
                "family": item.family,
                "components": list(item.components),
                "prompt_path": (
                    Path("semantic-interventions") / item.prompt_path
                ).as_posix(),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    bundle_hash = tree_hash(temporary)
    manifest = {
        "schema_version": "3.0",
        "semantic_protocol_version": SEMANTIC_PROTOCOL_VERSION,
        "created_at": utc_now(),
        "scope": "one_campaign_wide_multi_arm_semantic_prompt_bundle",
        "framework_id": framework.framework_key,
        "prompt_profile": framework.prompt_profile,
        "assumption_prompt_path": None,
        "bundle_sha256": bundle_hash,
        "interventions": interventions,
        "files": _file_manifest(temporary),
    }
    atomic_json(temporary / "manifest.json", manifest)
    os.replace(temporary, destination)
    for run_dir in sorted((campaign / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        run_manifest_path = run_dir / "manifest.json"
        run_manifest = _read_object(run_manifest_path)
        run_manifest["campaign_prompt_bundle_sha256"] = bundle_hash
        run_manifest["campaign_prompt_bundle"] = V3_PROMPT_BUNDLE.as_posix()
        atomic_json(run_manifest_path, run_manifest)
    campaign_manifest = _read_object(campaign / "campaign.json")
    campaign_manifest["campaign_prompt_bundle_sha256"] = bundle_hash
    atomic_json(campaign / "campaign.json", campaign_manifest)
    append_jsonl(
        campaign / "campaign-lifecycle.jsonl",
        {
            "schema_version": SEMANTIC_PROTOCOL_VERSION,
            "event": "semantic_prompt_bundle_snapshotted",
            "timestamp": utc_now(),
            "bundle_sha256": bundle_hash,
            "intervention_count": len(interventions),
        },
    )
    return manifest


def create_semantic_campaign(
    output_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    intervention_plan_path: str | Path,
    calibration_path: str | Path,
    repo_root: Path,
) -> Path:
    """Create a ready multi-arm campaign without starting any subject call."""

    plan = load_intervention_plan(intervention_plan_path)
    if spec.protocol_version != "3.0" or not spec.paired_prefix:
        raise ValueError("semantic v4 uses the unified v3 controller contract")
    if spec.blocks != plan.replicates:
        raise ValueError("protocol blocks must equal semantic replicates")
    if spec.first_fork_opportunity != plan.shared_prefix_opportunities + 1:
        raise ValueError("first intervention must immediately follow shared prefix")
    expected_schedule = tuple(
        range(
            plan.shared_prefix_opportunities + 1,
            spec.budget.proposals + 1,
            plan.session_span_opportunities,
        )
    )
    if spec.transition_opportunities != expected_schedule:
        raise ValueError("semantic interventions must align with phase starts")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    common_support = output / "task-support"
    prepare_seed_workspace(task, common_support, repo_root=repo_root)
    seed_id, common_snapshot = snapshot_candidate(
        common_support, output / "seed-candidate", task.editable_paths
    )
    calibration = _load_calibration(
        Path(calibration_path),
        spec=spec,
        task=task,
        candidate_id=seed_id,
        support_hash=tree_hash(common_support),
    )
    schedule = _schedule(spec=spec, task=task, framework=framework, plan=plan)
    runtime_hash = scientific_runtime_hash(repo_root, task=task, framework=framework)
    prompt_by_id = {
        item.intervention_id: (
            Path("semantic-interventions") / item.prompt_path
        ).as_posix()
        for item in plan.interventions
    }
    for assignment in schedule:
        run_id = str(assignment["run_id"])
        intervention_id = str(assignment["condition"])
        run_dir = output / "runs" / run_id
        seed = Candidate(
            candidate_id=seed_id,
            parent_ids=[],
            fitness=float(calibration["fitness"]),
            metrics=dict(calibration["metrics"]),
            artifact_path=f"candidates/{seed_id}",
            hypothesis="starting design",
            intended_edit="none",
            created_opportunity=0,
            retained_order=0,
        )
        SearchController.create(
            run_dir,
            spec,
            run_id=run_id,
            condition=Condition.C1,
            seed_candidate=seed,
        )
        shutil.copytree(common_support, run_dir / "task-support")
        destination = run_dir / "candidates" / seed_id
        destination.parent.mkdir()
        shutil.copytree(common_snapshot, destination)
        atomic_json(
            run_dir / "manifest.json",
            {
                "schema_version": SEMANTIC_PROTOCOL_VERSION,
                "assignment": assignment,
                "protocol_hash": spec.protocol_hash,
                "task_hash": sha256_json(task_hash_payload(task)),
                "framework_hash": sha256_json(framework_hash_payload(framework)),
                "scientific_runtime_hash": runtime_hash,
                "baseline": calibration,
                "repo_revision": _repo_revision(repo_root),
                "semantic_intervention": {
                    "id": intervention_id,
                    "label": assignment["condition_label"],
                    "family": assignment["condition_family"],
                    "components": assignment["components"],
                    "prompt_path": prompt_by_id[intervention_id],
                    "opportunities": list(spec.transition_opportunities),
                },
            },
        )
    inputs = output / "inputs"
    inputs.mkdir()
    atomic_json(inputs / "protocol.json", asdict(spec))
    atomic_json(inputs / "task.json", asdict(task))
    atomic_json(inputs / "framework.json", asdict(framework))
    shutil.copy2(intervention_plan_path, inputs / "semantic-interventions.toml")
    atomic_json(output / "schedule.json", schedule)
    atomic_json(
        output / "campaign.json",
        {
            "schema_version": SEMANTIC_PROTOCOL_VERSION,
            "design": "multi_arm_semantic_interventions_with_shared_prefix_v1",
            "study_id": spec.study_id,
            "search_architecture": framework.framework_key,
            "protocol_hash": spec.protocol_hash,
            "task_hash": sha256_json(task_hash_payload(task)),
            "framework_hash": sha256_json(framework_hash_payload(framework)),
            "scientific_runtime_hash": runtime_hash,
            "seed_candidate_id": seed_id,
            "support_tree_sha256": tree_hash(common_support),
            "intervention_plan_sha256": hashlib.sha256(
                Path(intervention_plan_path).read_bytes()
            ).hexdigest(),
            "replicates": plan.replicates,
            "intervention_count": len(plan.interventions),
            "scheduled_runs": len(schedule),
            "proposals_per_run": spec.budget.proposals,
            "shared_prefix_opportunities": plan.shared_prefix_opportunities,
            "session_span_opportunities": plan.session_span_opportunities,
            "analysis_unit": "complete_trajectory",
            "created_at": utc_now(),
        },
    )
    initialize_runtime_options(output)
    runtime = load_runtime_options(output)
    runtime["conversation"].update(
        {
            "mode": "five_opportunity_phased_sessions",
            "session_span_opportunities": plan.session_span_opportunities,
            "include_raw_transcript": False,
            "recent_lineage_outcomes": 6,
            "informative_failure_items": 4,
            "evidence_item_limit": 12,
            "evidence_character_limit": 28000,
        }
    )
    if task.task_id.startswith("fashion_mnist"):
        multi_fidelity = {
            "enabled": True,
            "strategy": "in_process_successive_screen_then_full_confirmation_v1",
            "training_examples": [25000, 50000, 100000],
            "command_argument": "--training-examples",
            "promotion_validation_accuracy": [0.82, 0.87, None],
            "full_fidelity_required_for_retention": True,
            "single_training_trajectory": True,
            "candidate_editable_policy": {
                "enabled": True,
                "path": "train.py",
                "levels_symbol": "EVALUATION_LADDER",
                "thresholds_symbol": "EVALUATION_PROMOTION_THRESHOLDS",
                "minimum_level": 10000,
                "maximum_level": 100000,
                "required_terminal_level": 100000,
                "max_rungs": 6,
            },
        }
    elif "adderboard-training-ladder" in task.task_id:
        multi_fidelity = {
            "enabled": True,
            "strategy": "escalate_until_qualified_v1",
            "levels": [5000, 10000, 15000, 20000, 25000, 30000],
            "command_argument": "--training-steps",
            "full_fidelity_required_for_retention": False,
            "stop_at_first_qualification": True,
            "candidate_editable_policy": {
                "enabled": True,
                "path": "src/train.py",
                "levels_symbol": "EVALUATION_LADDER",
                "minimum_level": 1000,
                "maximum_level": 30000,
                "required_terminal_level": 30000,
                "max_rungs": 10,
            },
        }
    else:
        multi_fidelity = {"enabled": False}
    runtime["evaluation"].update(
        {
            "task_pool_capacity": plan.task_evaluator_capacity,
            "multi_fidelity": multi_fidelity,
        }
    )
    runtime["developmental_reward"] = {
        "enabled": True,
        "selection_effect": "none",
        "visible_to_subject": True,
        "archive_capacity": 8,
        "valid_execution_credit": 0.25,
        "novel_delta_credit": 0.25,
        "near_incumbent_credit": 0.25,
        "retained_credit": 0.25,
        "near_incumbent_relative_margin": 0.02,
        "near_qualification_absolute_margin": 0.02,
    }
    runtime["semantic_interventions"] = {
        "protocol_version": SEMANTIC_PROTOCOL_VERSION,
        "active_at_phase_starts": True,
        "shared_prefix_opportunities": plan.shared_prefix_opportunities,
        "intervention_ids": list(plan.intervention_ids),
        "max_parallel_agent_calls": plan.max_parallel_agent_calls,
    }
    update_runtime_options(
        output,
        replacement=runtime,
        reason="initialize semantic intervention campaign controls",
    )
    prompt_manifest = _snapshot_semantic_prompts(
        output,
        spec=spec,
        framework=framework,
        plan=plan,
        repo_root=repo_root,
    )
    prefix_rows = []
    for replicate in range(1, plan.replicates + 1):
        members = [row for row in schedule if int(row["replicate"]) == replicate]
        leader = next(row for row in members if row["condition"] == "passive_control")
        prefix_rows.append(
            {
                "replicate": replicate,
                "leader_run_id": leader["run_id"],
                "shadow_run_ids": [
                    row["run_id"] for row in members if row is not leader
                ],
                "shared_through_opportunity": plan.shared_prefix_opportunities,
                "fork_opportunity": plan.shared_prefix_opportunities + 1,
                "resource_accounting": "shared_prefix_charge_once_per_replicate",
            }
        )
    atomic_json(
        output / SEMANTIC_PREFIX,
        {
            "schema_version": SEMANTIC_PROTOCOL_VERSION,
            "semantics": "literal_shared_prefix_then_multi_arm_fork",
            "replicates": prefix_rows,
        },
    )
    atomic_json(
        output / SEMANTIC_MANIFEST,
        {
            "schema_version": SEMANTIC_PROTOCOL_VERSION,
            "plan": asdict(plan),
            "prompt_bundle_sha256": prompt_manifest["bundle_sha256"],
        },
    )
    atomic_json(
        output / SEMANTIC_CONTROL,
        {
            "schema_version": SEMANTIC_PROTOCOL_VERSION,
            "desired": "paused",
            "reason": "prepared but not started",
            "updated_at": utc_now(),
        },
    )
    atomic_json(
        output / SEMANTIC_RUN_CONTROL,
        {
            "schema_version": SEMANTIC_PROTOCOL_VERSION,
            "runs": {
                str(row["run_id"]): {
                    "desired": "running",
                    "reason": "prepared for independent scheduling",
                    "updated_at": utc_now(),
                }
                for row in schedule
            },
        },
    )
    shutil.rmtree(common_support)
    shutil.rmtree(output / "seed-candidate")
    return output


def load_semantic_campaign(
    campaign: str | Path,
) -> tuple[Path, FactorialSpec, TaskSpec, FrameworkSpec, InterventionPlan]:
    root = Path(campaign).resolve()
    manifest = _read_object(root / "campaign.json")
    if manifest.get("schema_version") != SEMANTIC_PROTOCOL_VERSION:
        raise ValueError("not a semantic intervention v4 campaign")
    spec = (
        FactorialSpec.from_toml(root / "inputs/semantic-protocol.toml")
        if (root / "inputs/semantic-protocol.toml").is_file()
        else _spec_from_json(root / "inputs/protocol.json")
    )
    task = _task_from_json(root / "inputs/task.json")
    framework = _framework_from_json(root / "inputs/framework.json")
    plan = load_intervention_plan(root / "inputs/semantic-interventions.toml")
    return root, spec, task, framework, plan


def _spec_from_json(path: Path) -> FactorialSpec:
    value = _read_object(path)
    from .spec import BudgetSpec, ConversationMode, ModelSpec

    value["model"] = ModelSpec(**value["model"])
    value["budget"] = BudgetSpec(**value["budget"])
    value["conversation_mode"] = ConversationMode(value["conversation_mode"])
    value["transition_opportunities"] = tuple(value["transition_opportunities"])
    return FactorialSpec(**value)


def _task_from_json(path: Path) -> TaskSpec:
    value = _read_object(path)
    from .spec import ExecutionBackend, ObjectiveDirection

    value["editable_paths"] = tuple(value["editable_paths"])
    value["evaluator_command"] = tuple(value["evaluator_command"])
    value["public_feedback_metrics"] = tuple(value["public_feedback_metrics"])
    value["final_holdout_command"] = tuple(value["final_holdout_command"])
    value["objective_direction"] = ObjectiveDirection(value["objective_direction"])
    value["preferred_backend"] = ExecutionBackend(value["preferred_backend"])
    return TaskSpec(**value)


def _framework_from_json(path: Path) -> FrameworkSpec:
    value = _read_object(path)
    from .spec import FrameworkKind

    try:
        value["framework_id"] = FrameworkKind(value["framework_id"])
    except ValueError:
        value["framework_id"] = str(value["framework_id"])
    return FrameworkSpec(**value)


def validate_semantic_campaign(
    campaign: str | Path, *, repo_root: Path
) -> dict[str, Any]:
    root, spec, task, framework, plan = load_semantic_campaign(campaign)
    errors: list[str] = []
    try:
        validate_prompt_bundle(root, spec=spec, framework=framework)
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(str(error))
    schedule = json.loads((root / "schedule.json").read_text(encoding="utf-8"))
    expected_runs = plan.replicates * len(plan.interventions)
    if len(schedule) != expected_runs:
        errors.append(f"expected {expected_runs} scheduled runs")
    counts = Counter(str(row.get("condition")) for row in schedule)
    if any(
        counts.get(item.intervention_id) != plan.replicates
        for item in plan.interventions
    ):
        errors.append("every semantic arm must have exactly the configured replicates")
    try:
        run_desired = _run_desired_states(root)
        scheduled_ids = {str(row["run_id"]) for row in schedule}
        if set(run_desired) != scheduled_ids:
            errors.append("semantic run-control registry does not match the schedule")
        if any(
            value not in {"running", "paused", "stopped"}
            for value in run_desired.values()
        ):
            errors.append("semantic run-control registry has an invalid desired state")
    except (OSError, ValueError) as error:
        errors.append(f"semantic run-control registry: {error}")
    for replicate in range(1, plan.replicates + 1):
        seeds = {
            int(row["run_seed"])
            for row in schedule
            if int(row["replicate"]) == replicate
        }
        if len(seeds) != 1:
            errors.append(f"replicate {replicate} does not use one paired run seed")
    support_hashes = set()
    for row in schedule:
        run_dir = root / "runs" / str(row["run_id"])
        try:
            controller = SearchController.load(run_dir, spec)
            if controller.state.condition != Condition.C1.value:
                errors.append(
                    f"{run_dir.name} lacks the semantic campaign's C1 wrapper"
                )
            if (
                controller.state.proposals_used != 0
                or controller.state.active is not None
            ):
                errors.append(f"{run_dir.name} is not launch-ready")
            run_manifest = _read_object(run_dir / "manifest.json")
            if run_manifest.get("assignment") != row:
                errors.append(f"{run_dir.name} assignment mismatch")
            if run_manifest.get("scientific_runtime_hash") != scientific_runtime_hash(
                repo_root, task=task, framework=framework
            ):
                errors.append(f"{run_dir.name} scientific runtime mismatch")
            support_hashes.add(tree_hash(run_dir / "task-support"))
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(f"{run_dir.name}: {error}")
    if len(support_hashes) != 1:
        errors.append("run task-support trees are not byte-identical")
    prefix = _read_object(root / SEMANTIC_PREFIX)
    if len(prefix.get("replicates", [])) != plan.replicates:
        errors.append("semantic prefix manifest is incomplete")
    return {
        "schema_version": SEMANTIC_PROTOCOL_VERSION,
        "valid": not errors,
        "scheduled_runs": len(schedule),
        "interventions": len(plan.interventions),
        "replicates": plan.replicates,
        "physical_prefix_calls": plan.replicates * plan.shared_prefix_opportunities,
        "planned_postfork_calls": expected_runs
        * (spec.budget.proposals - plan.shared_prefix_opportunities),
        "errors": errors,
    }


def _prefix_for_run(campaign: Path, run_id: str) -> dict[str, Any]:
    value = _read_object(campaign / SEMANTIC_PREFIX)
    for row in value["replicates"]:
        if run_id == row["leader_run_id"] or run_id in row["shadow_run_ids"]:
            return dict(row)
    raise KeyError(f"run is absent from semantic prefix manifest: {run_id}")


@contextmanager
def _prefix_lock(campaign: Path, replicate: int) -> Iterator[None]:
    path = campaign / f".semantic-prefix-r{replicate:02d}.lock"
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _copy_prefix_to_shadow(
    campaign: Path,
    *,
    spec: FactorialSpec,
    leader_run_id: str,
    shadow_run_id: str,
    opportunity: int,
) -> None:
    leader_dir = campaign / "runs" / leader_run_id
    shadow_dir = campaign / "runs" / shadow_run_id
    leader = SearchController.load(leader_dir, spec)
    shadow = SearchController.load(shadow_dir, spec)
    if shadow.state.proposals_used == opportunity:
        return
    if shadow.state.proposals_used != opportunity - 1:
        raise RuntimeError("semantic shadow is not exactly one prefix step behind")
    source_opportunity = leader_dir / "opportunities" / f"{opportunity:04d}"
    target_opportunity = shadow_dir / "opportunities" / f"{opportunity:04d}"
    if not target_opportunity.exists():
        temporary = target_opportunity.with_name(
            f".{target_opportunity.name}.semantic-prefix-{os.getpid()}"
        )
        shutil.copytree(source_opportunity, temporary)
        os.replace(temporary, target_opportunity)
    for identifier in leader.state.candidates:
        source = leader_dir / "candidates" / identifier
        target = shadow_dir / "candidates" / identifier
        if source.is_dir() and not target.exists():
            shutil.copytree(source, target)
    framework = _framework_from_json(campaign / "inputs/framework.json")
    if is_native_openevolve(framework):
        mirror_native_prefix_state(leader_dir, shadow_dir)
    existing = {
        str(json.loads(line).get("source_event_sha256"))
        for line in (shadow_dir / "events.jsonl").read_text().splitlines()
        if json.loads(line).get("shared_prefix")
    }
    for line in (leader_dir / "events.jsonl").read_text().splitlines():
        record = json.loads(line)
        if int(record.get("opportunity", -1)) != opportunity:
            continue
        source_hash = _json_hash(record)
        if source_hash in existing:
            continue
        record["run_id"] = shadow_run_id
        record["shared_prefix"] = True
        record["shared_prefix_source_run_id"] = leader_run_id
        record["resource_accounting"] = "shared_prefix_charge_once_per_replicate"
        record["source_event_sha256"] = source_hash
        append_jsonl(shadow_dir / "events.jsonl", record)
    for result_name in ("result.json", "recovery.json"):
        result_path = target_opportunity / result_name
        if result_path.is_file():
            result = _read_object(result_path)
            result["run_id"] = shadow_run_id
            result["shared_prefix"] = True
            result["shared_prefix_source_run_id"] = leader_run_id
            result["resource_accounting"] = "shared_prefix_charge_once_per_replicate"
            atomic_json(result_path, result)
    state = leader.state.to_dict()
    state["run_id"] = shadow_run_id
    state["conversation_session_id"] = None
    state["revision"] = int(shadow.state.revision) + 1
    atomic_json(shadow_dir / "state.json", state)


def _mirror_prefix_completion(
    campaign: Path,
    *,
    spec: FactorialSpec,
    pair: dict[str, Any],
    opportunity: int,
    recovered: bool = False,
) -> None:
    """Mirror one charged prefix outcome and record it exactly once."""

    leader_run_id = str(pair["leader_run_id"])
    for shadow_id in pair["shadow_run_ids"]:
        _copy_prefix_to_shadow(
            campaign,
            spec=spec,
            leader_run_id=leader_run_id,
            shadow_run_id=str(shadow_id),
            opportunity=opportunity,
        )
    events_path = campaign / "semantic-prefix-events.jsonl"
    already_recorded = False
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if (
                event.get("event") == "shared_prefix_opportunity_completed"
                and int(event.get("replicate", -1)) == int(pair["replicate"])
                and int(event.get("opportunity", -1)) == opportunity
            ):
                already_recorded = True
                break
    if not already_recorded:
        append_jsonl(
            events_path,
            {
                "schema_version": SEMANTIC_PROTOCOL_VERSION,
                "event": "shared_prefix_opportunity_completed",
                "timestamp": utc_now(),
                "replicate": pair["replicate"],
                "opportunity": opportunity,
                "leader_run_id": leader_run_id,
                "inheriting_runs": len(pair["shadow_run_ids"]),
                "recovered_interruption": recovered,
                "resource_accounting": "shared_prefix_charge_once_per_replicate",
            },
        )


def run_semantic_opportunity(
    campaign: str | Path,
    *,
    run_id: str,
    repo_root: Path,
    python_bin: str,
    codex_binary: str = "codex",
    codex_timeout_seconds: int = 3600,
) -> dict[str, Any]:
    root, spec, task, framework, _plan = load_semantic_campaign(campaign)
    pair = _prefix_for_run(root, run_id)
    with _prefix_lock(root, int(pair["replicate"])):
        leader_id = str(pair["leader_run_id"])
        leader = SearchController.load(root / "runs" / leader_id, spec)
        requested = SearchController.load(root / "runs" / run_id, spec)
        fork = int(pair["fork_opportunity"])
        if requested.state.proposals_used < fork - 1:
            if leader.state.proposals_used == requested.state.proposals_used:
                record = run_one_opportunity(
                    root / "runs" / leader_id,
                    spec=spec,
                    task=task,
                    framework=framework,
                    repo_root=repo_root,
                    python_bin=python_bin,
                    codex_binary=codex_binary,
                    codex_timeout_seconds=codex_timeout_seconds,
                    allow_v3_prefix_leader=True,
                )
                opportunity = int(record["opportunity"])
            elif leader.state.proposals_used == requested.state.proposals_used + 1:
                opportunity = leader.state.proposals_used
                record = {"opportunity": opportunity, "recovered_pending_mirror": True}
            else:
                raise RuntimeError("semantic prefix states diverged before fork")
            _mirror_prefix_completion(
                root,
                spec=spec,
                pair=pair,
                opportunity=opportunity,
            )
            return {
                **record,
                "requested_run_id": run_id,
                "physical_run_id": leader_id,
                "shared_prefix": True,
            }
    record = run_one_opportunity(
        root / "runs" / run_id,
        spec=spec,
        task=task,
        framework=framework,
        repo_root=repo_root,
        python_bin=python_bin,
        codex_binary=codex_binary,
        codex_timeout_seconds=codex_timeout_seconds,
    )
    assignment = _read_object(root / "runs" / run_id / "manifest.json")["assignment"]
    record["semantic_intervention_id"] = assignment["condition"]
    record["semantic_intervention_family"] = assignment["condition_family"]
    if int(record["opportunity"]) in spec.transition_opportunities:
        append_jsonl(
            root / "runs" / run_id / "events.jsonl",
            {
                "schema_version": SEMANTIC_PROTOCOL_VERSION,
                "event": "semantic_intervention_applied",
                "timestamp": utc_now(),
                "run_id": run_id,
                "opportunity": record["opportunity"],
                "intervention_id": assignment["condition"],
                "intervention_family": assignment["condition_family"],
                "components": assignment["components"],
                "proposal_policy_sha256": record.get("prompt_hashes", {}).get(
                    "proposal_policy_sha256"
                ),
            },
        )
    atomic_json(
        root
        / "runs"
        / run_id
        / "opportunities"
        / f"{int(record['opportunity']):04d}"
        / "result.json",
        record,
    )
    return record


def set_semantic_control(
    campaign: str | Path, *, desired: str, reason: str
) -> dict[str, Any]:
    if desired not in {"running", "paused", "stopped"}:
        raise ValueError("semantic desired state must be running, paused, or stopped")
    if not reason.strip():
        raise ValueError("semantic control reason cannot be blank")
    root = Path(campaign).resolve()
    value = {
        "schema_version": SEMANTIC_PROTOCOL_VERSION,
        "desired": desired,
        "reason": reason.strip(),
        "updated_at": utc_now(),
    }
    atomic_json(root / SEMANTIC_CONTROL, value)
    append_jsonl(
        root / "semantic-lifecycle.jsonl", value | {"event": "desired_state_changed"}
    )
    return value


@contextmanager
def _run_control_lock(campaign: Path) -> Iterator[None]:
    path = campaign / ".semantic-run-control.lock"
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def set_semantic_run_control(
    campaign: str | Path, *, run_id: str, desired: str, reason: str
) -> dict[str, Any]:
    """Change one arm without perturbing any other scheduled trajectory."""

    if desired not in {"running", "paused", "stopped"}:
        raise ValueError("semantic run state must be running, paused, or stopped")
    if not reason.strip():
        raise ValueError("semantic run control reason cannot be blank")
    root = Path(campaign).resolve()
    with _run_control_lock(root):
        control = _read_object(root / SEMANTIC_RUN_CONTROL)
        runs = dict(control.get("runs", {}))
        if run_id not in runs:
            raise KeyError(f"unknown semantic run: {run_id}")
        value = {
            "desired": desired,
            "reason": reason.strip(),
            "updated_at": utc_now(),
        }
        runs[run_id] = value
        atomic_json(
            root / SEMANTIC_RUN_CONTROL,
            {
                "schema_version": SEMANTIC_PROTOCOL_VERSION,
                "runs": runs,
            },
        )
    append_jsonl(
        root / "semantic-lifecycle.jsonl",
        {
            "schema_version": SEMANTIC_PROTOCOL_VERSION,
            "event": "run_desired_state_changed",
            "timestamp": utc_now(),
            "run_id": run_id,
            **value,
        },
    )
    return {"run_id": run_id, **value}


def _run_desired_states(campaign: Path) -> dict[str, str]:
    control = _read_object(campaign / SEMANTIC_RUN_CONTROL)
    rows = control.get("runs")
    if not isinstance(rows, dict):
        raise ValueError("semantic run-control registry is malformed")
    return {
        str(run_id): str(value.get("desired"))
        for run_id, value in rows.items()
        if isinstance(value, dict)
    }


@contextmanager
def _orchestrator_lock(campaign: Path) -> Iterator[None]:
    path = campaign / ".semantic-orchestrator.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "semantic campaign already has an orchestrator"
            ) from error
        yield


def _resolve_worker_limit(requested: int | None, configured: int) -> int | None:
    """Return ``None`` for unbounded concurrency; zero is the public sentinel."""

    value = configured if requested is None else requested
    if value < 0:
        raise ValueError("max_workers must be nonnegative")
    return None if value == 0 else value


def _semantic_job_candidates(
    root: Path,
    *,
    schedule: list[dict[str, Any]],
    plan: InterventionPlan,
    states: dict[str, Any],
    run_desired: dict[str, str],
) -> list[tuple[str, bool]]:
    """Return independently runnable jobs and whether each is a prefix job."""

    jobs: list[tuple[str, bool]] = []
    prefix_manifest = _read_object(root / SEMANTIC_PREFIX)
    for prefix in prefix_manifest["replicates"]:
        leader_id = str(prefix["leader_run_id"])
        members = [leader_id, *map(str, prefix["shadow_run_ids"])]
        if states[leader_id].proposals_used < plan.shared_prefix_opportunities and any(
            states[member].status != "completed"
            and run_desired.get(member) == "running"
            for member in members
        ):
            jobs.append((leader_id, True))
    for row in schedule:
        run_id = str(row["run_id"])
        state = states[run_id]
        if state.status == "completed" or run_desired.get(run_id) != "running":
            continue
        if state.proposals_used < plan.shared_prefix_opportunities:
            continue
        jobs.append((run_id, False))
    return jobs


def run_semantic_campaign(
    campaign: str | Path,
    *,
    repo_root: Path,
    python_bin: str,
    max_workers: int | None = None,
    recover_interrupted: bool = False,
    codex_binary: str = "codex",
) -> dict[str, Any]:
    root, spec, task, framework, plan = load_semantic_campaign(campaign)
    worker_limit = _resolve_worker_limit(max_workers, plan.max_parallel_agent_calls)
    schedule = json.loads((root / "schedule.json").read_text(encoding="utf-8"))
    with _orchestrator_lock(root):
        preload_framework_runtime(framework, repo_root=repo_root)
        if recover_interrupted:
            for row in schedule:
                run_dir = root / "runs" / str(row["run_id"])
                state = SearchController.load(run_dir, spec).state
                if state.active is not None:
                    recovered = recover_active_opportunity(
                        run_dir,
                        spec=spec,
                        reason=(
                            "semantic orchestrator was interrupted after "
                            "opportunity start"
                        ),
                    )
                    opportunity = int(recovered["opportunity"])
                    pair = _prefix_for_run(root, str(row["run_id"]))
                    if (
                        str(row["run_id"]) == str(pair["leader_run_id"])
                        and opportunity <= plan.shared_prefix_opportunities
                    ):
                        with _prefix_lock(root, int(pair["replicate"])):
                            _mirror_prefix_completion(
                                root,
                                spec=spec,
                                pair=pair,
                                opportunity=opportunity,
                                recovered=True,
                            )
        completed_calls = 0
        worker_capacity = min(worker_limit or len(schedule), len(schedule))
        schedule_order = {
            str(row["run_id"]): index for index, row in enumerate(schedule)
        }
        last_dispatched = {run_id: -1 for run_id in schedule_order}
        dispatch_sequence = 0
        stopping_for: str | None = None
        futures: dict[Future[dict[str, Any]], tuple[str, bool]] = {}

        with ThreadPoolExecutor(max_workers=worker_capacity) as pool:
            while True:
                desired = str(_read_object(root / SEMANTIC_CONTROL).get("desired"))
                if stopping_for is None and desired != "running":
                    stopping_for = desired

                states = {
                    str(row["run_id"]): SearchController.load(
                        root / "runs" / str(row["run_id"]), spec
                    ).state
                    for row in schedule
                }
                unfinished = [
                    row
                    for row in schedule
                    if states[str(row["run_id"])].status != "completed"
                ]
                active_run_ids = {run_id for run_id, _prefix in futures.values()}

                if stopping_for is None:
                    run_desired = _run_desired_states(root)
                    candidates = [
                        job
                        for job in _semantic_job_candidates(
                            root,
                            schedule=schedule,
                            plan=plan,
                            states=states,
                            run_desired=run_desired,
                        )
                        if job[0] not in active_run_ids
                    ]
                    candidates.sort(
                        key=lambda job: (
                            last_dispatched[job[0]],
                            schedule_order[job[0]],
                        )
                    )
                    available = worker_capacity - len(futures)
                    for run_id, is_prefix in candidates[:available]:
                        next_opportunity = states[run_id].proposals_used + 1
                        future = pool.submit(
                            run_semantic_opportunity,
                            root,
                            run_id=run_id,
                            repo_root=repo_root,
                            python_bin=python_bin,
                            codex_binary=codex_binary,
                        )
                        futures[future] = (run_id, is_prefix)
                        dispatch_sequence += 1
                        last_dispatched[run_id] = dispatch_sequence
                        append_jsonl(
                            root / SEMANTIC_WAVES,
                            {
                                "schema_version": SEMANTIC_PROTOCOL_VERSION,
                                "event": "semantic_dispatch_started",
                                "timestamp": utc_now(),
                                "run_id": run_id,
                                "opportunity": next_opportunity,
                                "shared_prefix": is_prefix,
                                "dispatch_sequence": dispatch_sequence,
                                "active_dispatches": len(futures),
                                "worker_limit": worker_limit,
                                "worker_capacity": worker_capacity,
                                "scheduling_policy": "independent_event_driven",
                            },
                        )

                if not futures:
                    if stopping_for is not None:
                        return {
                            "status": stopping_for,
                            "completed_physical_calls": completed_calls,
                        }
                    if not unfinished:
                        set_semantic_control(
                            root,
                            desired="stopped",
                            reason="all semantic trajectories completed",
                        )
                        return {
                            "status": "completed",
                            "completed_physical_calls": completed_calls,
                        }
                    return {
                        "status": "no-runnable-trajectories",
                        "completed_physical_calls": completed_calls,
                        "unfinished": len(unfinished),
                    }

                done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    run_id, is_prefix = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        message = f"{type(error).__name__}: {error}"
                        append_jsonl(
                            root / SEMANTIC_WAVES,
                            {
                                "schema_version": SEMANTIC_PROTOCOL_VERSION,
                                "event": "semantic_dispatch_failed",
                                "timestamp": utc_now(),
                                "run_id": run_id,
                                "error": message,
                            },
                        )
                        recovered_record: dict[str, object] | None = None
                        try:
                            failed_controller = SearchController.load(
                                root / "runs" / run_id, spec
                            )
                            if failed_controller.state.active is not None:
                                recovered_record = recover_active_opportunity(
                                    root / "runs" / run_id,
                                    spec=spec,
                                    reason=(
                                        "semantic worker raised an infrastructure "
                                        f"exception: {message}"
                                    ),
                                )
                        except (OSError, RuntimeError, ValueError) as recovery_error:
                            append_jsonl(
                                root / SEMANTIC_WAVES,
                                {
                                    "schema_version": SEMANTIC_PROTOCOL_VERSION,
                                    "event": "semantic_recovery_failed",
                                    "timestamp": utc_now(),
                                    "run_id": run_id,
                                    "error": (
                                        f"{type(recovery_error).__name__}: "
                                        f"{recovery_error}"
                                    ),
                                },
                            )
                        affected = [run_id]
                        if is_prefix:
                            prefix = _prefix_for_run(root, run_id)
                            if recovered_record is not None:
                                with _prefix_lock(root, int(prefix["replicate"])):
                                    _mirror_prefix_completion(
                                        root,
                                        spec=spec,
                                        pair=prefix,
                                        opportunity=int(
                                            recovered_record["opportunity"]
                                        ),
                                        recovered=True,
                                    )
                            affected = [
                                str(prefix["leader_run_id"]),
                                *map(str, prefix["shadow_run_ids"]),
                            ]
                        for affected_run in affected:
                            set_semantic_run_control(
                                root,
                                run_id=affected_run,
                                desired="paused",
                                reason=(
                                    "isolated after semantic worker infrastructure "
                                    f"failure in {run_id}: {message}"
                                ),
                            )
                        continue
                    completed_calls += 1
                    append_jsonl(
                        root / SEMANTIC_WAVES,
                        {
                            "schema_version": SEMANTIC_PROTOCOL_VERSION,
                            "event": "semantic_dispatch_completed",
                            "timestamp": utc_now(),
                            "run_id": run_id,
                            "opportunity": result.get("opportunity"),
                            "shared_prefix": bool(result.get("shared_prefix")),
                            "active_dispatches": len(futures),
                            "scheduling_policy": "independent_event_driven",
                        },
                    )


def semantic_status(campaign: str | Path) -> dict[str, Any]:
    root, spec, _task, _framework, plan = load_semantic_campaign(campaign)
    schedule = json.loads((root / "schedule.json").read_text(encoding="utf-8"))
    run_desired = _run_desired_states(root)
    rows = []
    for assignment in schedule:
        state = SearchController.load(
            root / "runs" / str(assignment["run_id"]), spec
        ).state
        incumbent = state.candidates[state.incumbent_id]
        archive_path = root / "runs" / state.run_id / "developmental-archive.json"
        archive = (
            _read_object(archive_path) if archive_path.is_file() else {"items": []}
        )
        rows.append(
            {
                "run_id": state.run_id,
                "replicate": assignment["replicate"],
                "intervention_id": assignment["condition"],
                "family": assignment["condition_family"],
                "status": state.status,
                "desired": run_desired.get(state.run_id, "unknown"),
                "active_opportunity": state.active.index if state.active else None,
                "proposals_used": state.proposals_used,
                "evaluations_used": state.evaluations_used,
                "total_tokens": state.usage.total_tokens,
                "best_fitness": incumbent.fitness,
                "best_metrics": incumbent.metrics,
                "developmental_archive_items": len(archive.get("items", [])),
            }
        )
    return {
        "schema_version": SEMANTIC_PROTOCOL_VERSION,
        "desired": _read_object(root / SEMANTIC_CONTROL),
        "interventions": len(plan.interventions),
        "replicates": plan.replicates,
        "runs": rows,
        "counts": dict(Counter(str(row["status"]) for row in rows)),
    }
