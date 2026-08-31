"""Fail-closed campaign and prompt-control launch audit."""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import candidate_hash, scientific_runtime_hash, tree_hash
from .neutral_task import (
    FASHION_MNIST_SOURCE_ONLY_SEED_PATHS,
    FASHION_MNIST_TASK_ADAPTER,
    NANOGPT_SOURCE_ONLY_SEED_PATHS,
    NANOGPT_TASK_ADAPTER,
    NEUTRAL_TASK_ADAPTER,
    PAIR_TOKEN_SANITIZED_SEED_PATHS,
    PAIR_TOKEN_SOURCE_ONLY_SEED_PATHS,
    PAIR_TOKEN_TASK_ADAPTER,
    PAIR_TOKEN_TASK_ADAPTER_V2,
    PAIR_TOKEN_TASK_ADAPTER_V3,
    SANITIZED_SEED_PATHS,
    SUBJECT_NEUTRAL_PROMPT_PROFILES,
    TINY_ADDERBOARD_SOURCE_ONLY_SEED_PATHS,
    TINY_ADDERBOARD_TASK_ADAPTER,
    validate_v15_pairing,
)
from .prompts import (
    PromptContext,
    PromptRenderer,
    VisibleCandidate,
    artifact_clean_assumption_prompt_source,
    neutral_disclosure_terms,
    treatment_skeleton,
)
from .spec import (
    INDIVIDUAL_EXECUTION_RULES,
    PARALLEL_EXECUTION_RULE,
    SERIAL_EXECUTION_RULE,
    STAGED_CONFINED_INDIVIDUAL_EXECUTION_RULE,
    STAGED_EXECUTION_RULES,
    STAGED_INDEPENDENT_EXECUTION_RULE,
    STAGED_PARALLEL_EXECUTION_RULE,
    Condition,
    ExecutionBackend,
    FactorialSpec,
    FrameworkSpec,
    TaskSpec,
    framework_hash_payload,
    sha256_json,
    task_hash_payload,
)
from .state import SearchController
from .v3 import prompt_renderer_paths


def hybrid_modal_pairing_is_frozen(*, protocol_version: str, task_adapter: str) -> bool:
    """Return whether a protocol/task pair prospectively defines Modal transport."""

    return protocol_version in {"2.0", "2.1", "3.0"} or (
        protocol_version == "1.7" and task_adapter == NANOGPT_TASK_ADAPTER
    )


def neutral_source_disclosure_terms(source: str) -> tuple[str, ...]:
    """Audit code without treating ordinary c0/c1/c2/c3 variables as labels."""

    return tuple(
        term
        for term in neutral_disclosure_terms(source)
        if term not in {"c0", "c1", "c2", "c3"}
    )


def validate_campaign(
    campaign_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path,
) -> dict[str, object]:
    campaign = Path(campaign_dir).resolve()
    manifest = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    try:
        validate_v15_pairing(
            protocol_version=spec.protocol_version,
            task_adapter=task.adapter,
            prompt_profile=framework.prompt_profile,
        )
    except ValueError as error:
        errors.append(str(error))
    expected_hashes = {
        "protocol_hash": spec.protocol_hash,
        "task_hash": sha256_json(task_hash_payload(task)),
        "framework_hash": sha256_json(framework_hash_payload(framework)),
    }
    current_runtime_hash = scientific_runtime_hash(
        repo_root, task=task, framework=framework
    )
    if spec.protocol_version != "3.0":
        expected_hashes["scientific_runtime_hash"] = current_runtime_hash
    for name, expected in expected_hashes.items():
        if manifest.get(name) != expected:
            errors.append(f"campaign {name} mismatch")
    if spec.execution_rule in {
        PARALLEL_EXECUTION_RULE,
        *STAGED_EXECUTION_RULES,
    } and task.preferred_backend not in {
        ExecutionBackend.LOCAL,
        ExecutionBackend.HYBRID_MODAL,
    }:
        errors.append(
            "parallel condition rounds currently require a local task backend"
        )
    if (
        task.preferred_backend is ExecutionBackend.HYBRID_MODAL
        and not hybrid_modal_pairing_is_frozen(
            protocol_version=spec.protocol_version,
            task_adapter=task.adapter,
        )
    ):
        errors.append(
            "hybrid Modal evaluation is not frozen for this protocol/task pairing"
        )
    schedule = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    run_ids = [str(row["run_id"]) for row in schedule]
    if len(run_ids) != len(set(run_ids)):
        errors.append("campaign schedule contains duplicate run IDs")
    expected_primary = (
        run_ids
        if spec.c0c3_only
        else [
            str(row["run_id"])
            for row in schedule
            if int(row["block"]) == 1 and str(row["condition"]) != "N0"
        ]
        if spec.execution_rule in STAGED_EXECUTION_RULES
        else run_ids
    )
    expected_optional = [
        run_id for run_id in run_ids if run_id not in set(expected_primary)
    ]
    if manifest.get("primary_run_ids") != expected_primary:
        errors.append("campaign primary run scope mismatch")
    if manifest.get("optional_run_ids") != expected_optional:
        errors.append("campaign optional run scope mismatch")
    if manifest.get("include_no_search") != spec.include_no_search:
        errors.append("campaign N0 composition differs from the frozen protocol")
    if (
        spec.execution_rule in STAGED_EXECUTION_RULES
        and not manifest.get("include_no_search")
        and not spec.c0c3_only
    ):
        errors.append("staged protocol requires pre-created N0 extensions")
    for block in range(1, spec.blocks + 1):
        rows = [row for row in schedule if int(row["block"]) == block]
        factorial = [row["condition"] for row in rows if row["condition"] != "N0"]
        if sorted(factorial) != sorted(condition.value for condition in Condition):
            errors.append(f"block {block} does not contain C0-C3 exactly once")
        if len({row["run_seed"] for row in rows}) != 1:
            errors.append(f"block {block} is not seed paired")
        expected_orders = list(range(1, len(rows) + 1))
        if sorted(int(row["order"]) for row in rows) != expected_orders:
            errors.append(f"block {block} execution order is not contiguous")
    support_hashes = set()
    seed_ids = set()
    launch_states: list[str] = []
    for assignment in schedule:
        run_dir = campaign / "runs" / str(assignment["run_id"])
        try:
            controller = SearchController.load(run_dir, spec)
            support_hashes.add(tree_hash(run_dir / "task-support"))
            seed_id = str(manifest["seed_candidate_id"])
            seed_ids.add(
                candidate_hash(run_dir / "candidates" / seed_id, task.editable_paths)
            )
            if controller.state.condition != assignment["condition"]:
                errors.append(f"{assignment['run_id']} condition mismatch")
            if (
                controller.state.status != "ready"
                or controller.state.proposals_used != 0
                or controller.state.evaluations_used != 0
                or controller.state.active is not None
            ):
                launch_states.append(str(assignment["run_id"]))
            run_manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            for name, expected in expected_hashes.items():
                if run_manifest.get(name) != expected:
                    errors.append(f"{assignment['run_id']} {name} mismatch")
            if run_manifest.get("assignment") != assignment:
                errors.append(f"{assignment['run_id']} assignment mismatch")
        except (FileNotFoundError, KeyError, ValueError) as error:
            errors.append(f"{assignment['run_id']} invalid: {error}")
    if len(support_hashes) != 1:
        errors.append("run task-support trees are not byte-identical")
    if task.adapter in {
        NEUTRAL_TASK_ADAPTER,
        PAIR_TOKEN_TASK_ADAPTER,
        PAIR_TOKEN_TASK_ADAPTER_V2,
        PAIR_TOKEN_TASK_ADAPTER_V3,
        NANOGPT_TASK_ADAPTER,
        FASHION_MNIST_TASK_ADAPTER,
        TINY_ADDERBOARD_TASK_ADAPTER,
    }:
        if task.adapter == NANOGPT_TASK_ADAPTER:
            sanitized_paths = NANOGPT_SOURCE_ONLY_SEED_PATHS
        elif task.adapter == FASHION_MNIST_TASK_ADAPTER:
            sanitized_paths = FASHION_MNIST_SOURCE_ONLY_SEED_PATHS
        elif task.adapter == TINY_ADDERBOARD_TASK_ADAPTER:
            sanitized_paths = TINY_ADDERBOARD_SOURCE_ONLY_SEED_PATHS
        elif task.adapter == NEUTRAL_TASK_ADAPTER:
            sanitized_paths = SANITIZED_SEED_PATHS
        elif task.adapter == PAIR_TOKEN_TASK_ADAPTER_V3:
            sanitized_paths = PAIR_TOKEN_SOURCE_ONLY_SEED_PATHS
        else:
            sanitized_paths = PAIR_TOKEN_SANITIZED_SEED_PATHS
        expected_subject_files = set(sanitized_paths)
        if task.adapter not in {
            NANOGPT_TASK_ADAPTER,
            FASHION_MNIST_TASK_ADAPTER,
            TINY_ADDERBOARD_TASK_ADAPTER,
        }:
            expected_subject_files.add("submission.py")
        actual_subject_files = {
            path.relative_to(campaign / "runs" / run_ids[0] / "task-support").as_posix()
            for path in (campaign / "runs" / run_ids[0] / "task-support").rglob("*")
            if path.is_file()
        }
        if actual_subject_files != expected_subject_files:
            errors.append("artifact-clean task-support tree exposes unexpected files")
        source_paths_to_audit = (
            ()
            if task.adapter == NANOGPT_TASK_ADAPTER
            else sorted(expected_subject_files)
        )
        for relative in source_paths_to_audit:
            if not relative.endswith(".py"):
                continue
            source = (
                campaign / "runs" / run_ids[0] / "task-support" / relative
            ).read_text(encoding="utf-8", errors="replace")
            disclosures = neutral_source_disclosure_terms(source)
            if disclosures:
                errors.append(
                    f"neutral task-support {relative} exposes {list(disclosures)}"
                )
    if seed_ids != {manifest.get("seed_candidate_id")}:
        errors.append("run seed candidates are not byte-identical")
    if launch_states:
        errors.append(
            "launch validation requires untouched ready runs: "
            + ", ".join(launch_states)
        )
    if spec.protocol_version == "3.0":
        try:
            template_root, transition_override = prompt_renderer_paths(
                campaign, spec=spec, framework=framework
            )
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(str(error))
            template_root = repo_root / "experiments/c0c3_factorial/templates"
            transition_override = None
    else:
        template_root = repo_root / "experiments/c0c3_factorial/templates"
        transition_override = artifact_clean_assumption_prompt_source(
            campaign=campaign,
            repo_root=repo_root,
            framework=framework,
        )
    renderer = PromptRenderer(
        template_root,
        artifact_clean_transition_override=transition_override,
    )
    for opportunity in range(1, spec.budget.proposals + 1):
        prompts = {}
        for condition in Condition:
            context = PromptContext(
                condition=condition,
                opportunity=opportunity,
                selected_parent_id="seed",
                visible_candidates=(
                    VisibleCandidate(
                        "seed", 0.0, {task.objective_metric: 0.0}, 0, "slot-1"
                    ),
                ),
                remaining_proposals=spec.budget.proposals,
                remaining_evaluations=spec.budget.candidate_evaluations,
                remaining_tokens=spec.budget.max_total_tokens,
                remaining_evaluator_seconds=spec.budget.max_evaluator_seconds,
            )
            prompts[condition] = renderer.render(spec, task, framework, context)
            if framework.prompt_profile in SUBJECT_NEUTRAL_PROMPT_PROFILES:
                disclosures = neutral_disclosure_terms(prompts[condition].text)
                if disclosures:
                    errors.append(
                        "neutral prompt exposes internal terms at opportunity "
                        f"{opportunity}: {list(disclosures)}"
                    )
        skeletons = {
            value.treatment_skeleton_sha256 or treatment_skeleton(value.text)
            for value in prompts.values()
        }
        if len(skeletons) != 1:
            errors.append(f"prompt skeleton differs at opportunity {opportunity}")
        if (
            prompts[Condition.C0].search_state_sha256
            != prompts[Condition.C1].search_state_sha256
            or prompts[Condition.C2].search_state_sha256
            != prompts[Condition.C3].search_state_sha256
        ):
            errors.append(f"search-state factor mismatch at opportunity {opportunity}")
        if (
            prompts[Condition.C0].proposal_policy_sha256
            != prompts[Condition.C2].proposal_policy_sha256
            or prompts[Condition.C1].proposal_policy_sha256
            != prompts[Condition.C3].proposal_policy_sha256
        ):
            errors.append(
                f"proposal-policy factor mismatch at opportunity {opportunity}"
            )
        if framework.prompt_profile in SUBJECT_NEUTRAL_PROMPT_PROFILES:
            if not spec.include_no_search:
                continue
            n0_context = PromptContext(
                condition=Condition.C0,
                opportunity=opportunity,
                selected_parent_id="seed",
                visible_candidates=(
                    VisibleCandidate(
                        "seed", 0.0, {task.objective_metric: 0.0}, 0, "design-1"
                    ),
                ),
                remaining_proposals=spec.budget.proposals,
                remaining_evaluations=spec.budget.candidate_evaluations,
                remaining_tokens=spec.budget.max_total_tokens,
                remaining_evaluator_seconds=spec.budget.max_evaluator_seconds,
                no_search=True,
            )
            n0_prompt = renderer.render(spec, task, framework, n0_context)
            disclosures = neutral_disclosure_terms(n0_prompt.text)
            if disclosures:
                errors.append(
                    "neutral N0 prompt exposes internal terms at opportunity "
                    f"{opportunity}: {list(disclosures)}"
                )
    layers_absent = not (
        (campaign / "sealed-layer-b").exists() or (campaign / "sealed-layer-c").exists()
    )
    if not layers_absent:
        errors.append("Layer B or Layer C output exists before launch")
    report = {
        "schema_version": "1.0",
        "campaign": str(campaign),
        "valid": not errors,
        "errors": errors,
        "checked_run_count": len(schedule),
        "protocol_hash": spec.protocol_hash,
        "controls": {
            "same_model_and_settings": True,
            "same_starting_artifact": len(seed_ids) == 1,
            "same_task_and_evaluator": len(support_hashes) == 1,
            "same_budget": True,
            "same_failure_rule": True,
            "execution_rule": spec.execution_rule,
            "frozen_execution_rule": (
                spec.execution_rule if spec.protocol_version != "3.0" else None
            ),
            "frozen_blocked_round_robin_execution": (
                spec.execution_rule == SERIAL_EXECUTION_RULE
            ),
            "frozen_parallel_condition_rounds": (
                spec.execution_rule == PARALLEL_EXECUTION_RULE
            ),
            "frozen_staged_parallel_trajectories": (
                spec.execution_rule == STAGED_PARALLEL_EXECUTION_RULE
            ),
            "frozen_staged_independent_trajectories": (
                spec.execution_rule == STAGED_INDEPENDENT_EXECUTION_RULE
            ),
            "frozen_staged_individually_controlled_trajectories": (
                spec.execution_rule in INDIVIDUAL_EXECUTION_RULES
            ),
            "confined_continuous_sessions": (
                spec.execution_rule == STAGED_CONFINED_INDIVIDUAL_EXECUTION_RULE
            ),
            "n0_removed_by_protocol": not spec.include_no_search,
            "layer_b_c_absent_at_launch": layers_absent,
        },
    }
    (campaign / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
