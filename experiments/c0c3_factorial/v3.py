"""Unified v3 controls shared by Autoresearch, OpenEvolve architectures, and plugins.

V3 deliberately separates one campaign-wide prompt snapshot from mutable
runtime/research options.  The prompt bundle is the only new immutable launch
artifact.  Every other v3 choice is explicit, versioned, and may be revised
with an append-only receipt so new tasks, adapters, and research variants do
not require another hard-coded protocol family.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .artifacts import scientific_runtime_hash, tree_hash
from .neutral_task import ARTIFACT_CLEAN_ASSUMPTION_PROMPT_PATHS
from .spec import (
    FactorialSpec,
    FrameworkSpec,
    TaskSpec,
    framework_hash_payload,
    sha256_json,
    task_hash_payload,
)
from .state import SearchController, append_jsonl, atomic_json, utc_now

V3_RUNTIME_FILE = Path("inputs/v3-runtime.json")
V3_RUNTIME_HISTORY = Path("v3-runtime-history.jsonl")
V3_PROMPT_BUNDLE = Path("prompt-bundle")
V3_PROMPT_MANIFEST = V3_PROMPT_BUNDLE / "manifest.json"
V3_PAIRING_FILE = Path("paired-prefix.json")


def default_runtime_options() -> dict[str, Any]:
    """Return flexible v3 controls and available research variants.

    These values are operational defaults, not immutable protocol gates.  A
    revision is allowed and is recorded by :func:`update_runtime_options`.
    """

    return {
        "schema_version": "3.0",
        "conversation": {
            "mode": "bounded_state_capsule",
            "session_span_opportunities": 1,
            "include_raw_transcript": False,
            "recent_lineage_outcomes": 4,
            "informative_failure_items": 3,
            "evidence_item_limit": 10,
            "evidence_character_limit": 24000,
            "agent_notes_character_limit": 4000,
        },
        "intervention": {
            "variant": "assumption_change_with_feasibility_and_falsification",
            "schedule_mode": "configured_indices",
            "cooldown_opportunities": 0,
            "alternative_variants_available": [
                "reflection_then_edit",
                "evidence_contradiction",
                "counterfactual_design",
                "constraint_inversion",
                "stagnation_triggered",
                "wording_robustness_library",
                "self_critique_control",
            ],
        },
        "portfolio": {
            "estimand": "whole_system",
            "evidence_selector": "informative_deterministic_v1",
            "parent_selector": "fair_lineage",
            "retention": "strict_selected_lineage_improvement",
            "source_representation": "full_source",
            "variants_available": [
                "matched_context_branch_diversity",
                "capacity_2_4_8",
                "fitness_proportional_selection",
                "uncertainty_aware_selection",
                "novelty_aware_selection",
                "minimum_quality_floor",
                "elite_plus_exploration",
                "pareto_retention",
                "similarity_deduplication",
                "branch_retirement",
                "cross_parent_recombination",
                "larger_archive_retrieval",
                "compact_diff_or_architecture_ir",
            ],
        },
        "evaluation": {
            "task_isolated_pool": True,
            "task_pool_capacity": 1,
            "shared_host_safety_ceiling": True,
            "queue_wait_counts_as_candidate_time": False,
            "paired_training_seeds": 1,
            "retention_confirmation": "public_single_seed",
            "multi_fidelity": {"enabled": False},
            "timeout_role": "safety_guard",
        },
        "remote": {
            "stable_call_ids": True,
            "dispatch_receipts": True,
            "retrieve_completed_result": True,
            "exactly_once_charge": True,
        },
        "analysis": {
            "manipulation_checks": True,
            "blinded_review_packets": True,
            "source_delta_authoritative": True,
            "mechanism_semantic_fingerprints": True,
            "trajectory_secondary_outcomes": True,
            "exploration_and_retention_outcomes": True,
            "paired_prefix_effect_contrast": True,
        },
        "instrumentation": {
            "priced_token_components": True,
            "phase_token_attribution": True,
            "queue_runtime_pause_sleep_wall_time": True,
            "environment_receipts": True,
            "candidate_diffs_and_architecture_ir": True,
            "health_status": True,
            "read_only_audit": True,
            "release_data_dictionary": True,
        },
        "companions": {
            "ecological_autoresearch_available": True,
            "native_openevolve_available": True,
            "model_and_task_replication_available": True,
        },
    }


def initialize_runtime_options(campaign: Path) -> Path:
    target = campaign / V3_RUNTIME_FILE
    atomic_json(target, default_runtime_options())
    append_jsonl(
        campaign / V3_RUNTIME_HISTORY,
        {
            "schema_version": "3.0",
            "event": "v3_runtime_initialized",
            "timestamp": utc_now(),
            "sha256": _json_hash(default_runtime_options()),
        },
    )
    atomic_json(
        campaign / "inputs/v3-pricing.json",
        {
            "schema_version": "3.0",
            "currency": "USD_list_price_equivalent",
            "effective_at": None,
            "models": {},
            "missing_price_behavior": "retain_raw_tokens_and_flag_unpriced",
            "token_components": [
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
            ],
        },
    )
    atomic_json(
        campaign / "environment-receipt.json",
        {
            "schema_version": "3.0",
            "recorded_at": utc_now(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "timezone": os.environ.get("TZ"),
            "cpu_count": os.cpu_count(),
        },
    )
    (campaign / "DATA_DICTIONARY.md").write_text(
        """# Unified v3 data dictionary

- `state.json`: crash-safe online search state for one trajectory.
- `events.jsonl`: proposal, accounting, and provenance events.
- `paired-prefix.json`: C0/C1 and C2/C3 prefix/fork relationships.
- `paired-prefix-events.jsonl`: inherited prefix operations; charge each pair once.
- `prompt-bundle/`: the campaign-wide prompt snapshot.
- `inputs/v3-runtime.json`: revisable v3 runtime and research options.
- `v3-runtime-history.jsonl`: append-only option revision receipts.
- `opportunities/NNNN/state-capsule.json`: bounded evidence boundary.
- `opportunities/NNNN/candidate-provenance.json`: source delta and architecture IR.
- `opportunities/NNNN/remote-dispatch.json`: idempotent remote-call receipt.
- `review/v3-manipulation/`: blinded checks and private condition mapping.
""",
        encoding="utf-8",
    )
    return target


def load_runtime_options(campaign: Path) -> dict[str, Any]:
    value = json.loads((campaign / V3_RUNTIME_FILE).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "3.0":
        raise ValueError("invalid v3 runtime options")
    return value


def update_runtime_options(
    campaign: Path, *, replacement: dict[str, Any], reason: str
) -> dict[str, Any]:
    """Replace flexible v3 options while preserving a complete audit trail."""

    if not reason.strip():
        raise ValueError("runtime revision reason cannot be blank")
    current = load_runtime_options(campaign)
    if replacement.get("schema_version") != "3.0":
        raise ValueError("replacement must retain schema_version='3.0'")
    record = {
        "schema_version": "3.0",
        "event": "v3_runtime_revised",
        "timestamp": utc_now(),
        "reason": reason.strip(),
        "before_sha256": _json_hash(current),
        "after_sha256": _json_hash(replacement),
        "before": current,
        "after": replacement,
    }
    atomic_json(campaign / V3_RUNTIME_FILE, replacement)
    append_jsonl(campaign / V3_RUNTIME_HISTORY, record)
    return record


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_manifest(root: Path) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def snapshot_prompt_bundle(
    campaign: Path,
    *,
    spec: FactorialSpec,
    framework: FrameworkSpec,
    repo_root: Path,
) -> dict[str, Any]:
    """Create the sole v3 frozen artifact: one campaign-wide prompt bundle."""

    if spec.protocol_version != "3.0":
        raise ValueError("campaign-wide v3 prompt snapshots require protocol 3.0")
    destination = campaign / V3_PROMPT_BUNDLE
    if destination.exists():
        raise FileExistsError("v3 campaign prompt bundle already exists")
    source = repo_root / "experiments/c0c3_factorial/templates"
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    shutil.copytree(source, temporary)
    assumption_relative = ARTIFACT_CLEAN_ASSUMPTION_PROMPT_PATHS.get(
        framework.prompt_profile
    )
    if assumption_relative is not None:
        assumption_path = temporary / assumption_relative
        existing = assumption_path.read_text(encoding="utf-8").rstrip()
        feasibility = (
            "\n\nIdentify the strongest evidence for and against the load-bearing "
            "assumption you choose. Use an implementation that decisively tests "
            "the alternative while remaining feasible under the task's training "
            "and model constraints. Preserve components supported by evidence when "
            "changing them is not needed to test the alternative. Explain what "
            "observed result would falsify the new approach.\n"
        )
        assumption_path.write_text(existing + feasibility, encoding="utf-8")
    files = _file_manifest(temporary)
    bundle_hash = tree_hash(temporary)
    manifest = {
        "schema_version": "3.0",
        "created_at": utc_now(),
        "scope": "one_campaign_wide_ordinary_and_assumption_prompt_bundle",
        "framework_id": framework.framework_key,
        "prompt_profile": framework.prompt_profile,
        "assumption_prompt_path": assumption_relative,
        "bundle_sha256": bundle_hash,
        "files": files,
    }
    atomic_json(temporary / "manifest.json", manifest)
    os.replace(temporary, destination)
    for run_dir in sorted((campaign / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        run_manifest_path = run_dir / "manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        run_manifest["campaign_prompt_bundle_sha256"] = bundle_hash
        run_manifest["campaign_prompt_bundle"] = V3_PROMPT_BUNDLE.as_posix()
        atomic_json(run_manifest_path, run_manifest)
    campaign_path = campaign / "campaign.json"
    campaign_manifest = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign_manifest["campaign_prompt_bundle_sha256"] = bundle_hash
    atomic_json(campaign_path, campaign_manifest)
    append_jsonl(
        campaign / "campaign-lifecycle.jsonl",
        {
            "schema_version": "3.0",
            "event": "campaign_prompt_bundle_snapshotted",
            "timestamp": utc_now(),
            "bundle_sha256": bundle_hash,
        },
    )
    return manifest


def validate_prompt_bundle(
    campaign: Path, *, spec: FactorialSpec, framework: FrameworkSpec
) -> dict[str, Any]:
    if spec.protocol_version != "3.0":
        return {}
    manifest_path = campaign / V3_PROMPT_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError("v3 campaign must snapshot its prompt bundle before launch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ignored = frozenset({"manifest.json"})
    actual = tree_hash(campaign / V3_PROMPT_BUNDLE, ignored_relative_paths=ignored)
    if actual != manifest.get("bundle_sha256"):
        raise RuntimeError("v3 campaign prompt bundle changed after snapshot")
    if manifest.get("framework_id") != framework.framework_key:
        raise RuntimeError("v3 prompt bundle framework does not match campaign")
    expected = manifest["bundle_sha256"]
    for run_dir in (campaign / "runs").iterdir():
        if not run_dir.is_dir():
            continue
        run_manifest = json.loads((run_dir / "manifest.json").read_text())
        if run_manifest.get("campaign_prompt_bundle_sha256") != expected:
            raise RuntimeError("not every v3 run points to the campaign prompt bundle")
    return manifest


def prompt_renderer_paths(
    campaign: Path, *, spec: FactorialSpec, framework: FrameworkSpec
) -> tuple[Path, Path | None]:
    """Resolve renderer paths, enforcing the v3 campaign-wide snapshot gate."""

    if spec.protocol_version != "3.0":
        raise ValueError("prompt_renderer_paths is a v3 helper")
    manifest = validate_prompt_bundle(campaign, spec=spec, framework=framework)
    template_root = campaign / V3_PROMPT_BUNDLE
    relative = manifest.get("assumption_prompt_path")
    override = template_root / str(relative) if relative else None
    return template_root, override


def create_pairing_manifest(campaign: Path, *, spec: FactorialSpec) -> Path:
    if spec.protocol_version != "3.0":
        raise ValueError("paired prefixes require protocol 3.0")
    schedule = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    pairs = []
    for block in range(1, spec.blocks + 1):
        by_condition = {
            str(row["condition"]): str(row["run_id"])
            for row in schedule
            if int(row["block"]) == block
        }
        for leader, shadow in (("C0", "C1"), ("C2", "C3")):
            pairs.append(
                {
                    "block": block,
                    "leader_condition": leader,
                    "leader_run_id": by_condition[leader],
                    "shadow_condition": shadow,
                    "shadow_run_id": by_condition[shadow],
                    "shared_through_opportunity": spec.first_fork_opportunity - 1,
                    "fork_opportunity": spec.first_fork_opportunity,
                    "resource_accounting": "shared_prefix_charge_once_to_pair",
                }
            )
    path = campaign / V3_PAIRING_FILE
    atomic_json(
        path,
        {
            "schema_version": "3.0",
            "semantics": "literal_shared_trajectory_then_identical_state_fork",
            "pairs": pairs,
        },
    )
    return path


def pair_for_run(campaign: Path, run_id: str) -> dict[str, Any]:
    payload = json.loads((campaign / V3_PAIRING_FILE).read_text(encoding="utf-8"))
    for pair in payload["pairs"]:
        if run_id in {pair["leader_run_id"], pair["shadow_run_id"]}:
            return pair
    raise KeyError(f"v3 pairing manifest has no pair for {run_id}")


@contextmanager
def pair_lock(campaign: Path, pair: dict[str, Any]) -> Iterator[None]:
    block = int(pair["block"])
    leader = str(pair["leader_condition"]).lower()
    path = campaign / f".v3-pair-b{block:02d}-{leader}.lock"
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def mirror_shared_prefix(
    campaign: Path,
    *,
    spec: FactorialSpec,
    pair: dict[str, Any],
) -> dict[str, Any]:
    """Copy one completed leader opportunity into its untreated shadow.

    The copy is scientific inheritance, not another model or evaluator call.
    It is therefore present in both trajectories and explicitly charge-once.
    """

    leader_dir = campaign / "runs" / str(pair["leader_run_id"])
    shadow_dir = campaign / "runs" / str(pair["shadow_run_id"])
    leader = SearchController.load(leader_dir, spec)
    shadow = SearchController.load(shadow_dir, spec)
    completed = leader.state.proposals_used
    if completed >= int(pair["fork_opportunity"]):
        raise ValueError("cannot mirror an opportunity at or after the v3 fork")
    if shadow.state.proposals_used != completed - 1:
        raise RuntimeError("v3 shadow is not exactly one opportunity behind leader")
    if shadow.state.active is not None or leader.state.active is not None:
        raise RuntimeError("cannot mirror an active v3 opportunity")

    opportunity = completed
    source_opportunity = leader_dir / "opportunities" / f"{opportunity:04d}"
    target_opportunity = shadow_dir / "opportunities" / f"{opportunity:04d}"
    if not target_opportunity.exists():
        temporary = target_opportunity.with_name(
            f".{target_opportunity.name}.paired-prefix-{os.getpid()}"
        )
        shutil.copytree(source_opportunity, temporary)
        os.replace(temporary, target_opportunity)
    for identifier in leader.state.candidates:
        source = leader_dir / "candidates" / identifier
        target = shadow_dir / "candidates" / identifier
        if source.is_dir() and not target.exists():
            shutil.copytree(source, target)
    framework_input = json.loads(
        (campaign / "inputs/framework.json").read_text(encoding="utf-8")
    )
    if framework_input.get("framework_id") == "native_openevolve":
        from .native_openevolve import mirror_native_prefix_state

        mirror_native_prefix_state(leader_dir, shadow_dir)

    records = []
    existing_shared_events = set()
    for line in (shadow_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        existing = json.loads(line)
        if existing.get("shared_prefix") and existing.get("opportunity") == opportunity:
            existing_shared_events.add(str(existing.get("source_event_sha256")))
    for line in (leader_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record.get("opportunity", -1)) != opportunity:
            continue
        record["run_id"] = str(pair["shadow_run_id"])
        record["condition"] = str(pair["shadow_condition"])
        record["shared_prefix"] = True
        record["shared_prefix_source_run_id"] = str(pair["leader_run_id"])
        record["resource_accounting"] = "shared_prefix_charge_once_to_pair"
        record["source_event_sha256"] = _json_hash(json.loads(line))
        if record["source_event_sha256"] not in existing_shared_events:
            append_jsonl(shadow_dir / "events.jsonl", record)
        records.append(record)
    result_path = target_opportunity / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["run_id"] = str(pair["shadow_run_id"])
        result["condition"] = str(pair["shadow_condition"])
        result["shared_prefix"] = True
        result["shared_prefix_source_run_id"] = str(pair["leader_run_id"])
        result["resource_accounting"] = "shared_prefix_charge_once_to_pair"
        atomic_json(result_path, result)

    # Commit state last. If the process dies earlier, the next paired call sees
    # the one-step lag and idempotently finishes the same mirror operation.
    state = leader.state.to_dict()
    state["run_id"] = str(pair["shadow_run_id"])
    state["condition"] = str(pair["shadow_condition"])
    state["conversation_session_id"] = None
    state["revision"] = int(shadow.state.revision) + 1
    atomic_json(shadow_dir / "state.json", state)
    append_jsonl(
        campaign / "paired-prefix-events.jsonl",
        {
            "schema_version": "3.0",
            "event": "shared_prefix_opportunity_mirrored",
            "timestamp": utc_now(),
            "block": pair["block"],
            "opportunity": opportunity,
            "leader_run_id": pair["leader_run_id"],
            "shadow_run_id": pair["shadow_run_id"],
            "candidate_id": leader.state.incumbent_id,
            "resource_accounting": "shared_prefix_charge_once_to_pair",
        },
    )
    return {
        "opportunity": opportunity,
        "leader_run_id": pair["leader_run_id"],
        "shadow_run_id": pair["shadow_run_id"],
        "event_records": len(records),
    }


def v3_health_report(
    campaign: Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    framework: FrameworkSpec,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run a read-only, framework-neutral v3 launch/continuation health gate."""

    errors: list[str] = []
    checks: dict[str, Any] = {}
    try:
        manifest = validate_prompt_bundle(campaign, spec=spec, framework=framework)
        checks["prompt_bundle_sha256"] = manifest["bundle_sha256"]
    except (OSError, ValueError, RuntimeError) as error:
        errors.append(str(error))
    try:
        options = load_runtime_options(campaign)
        checks["runtime_options_sha256"] = _json_hash(options)
    except (OSError, ValueError) as error:
        errors.append(str(error))
    disk = shutil.disk_usage(campaign)
    checks["disk_free_bytes"] = disk.free
    if disk.free < 2 * 1024**3:
        errors.append("less than 2 GiB free disk space")
    run_dirs = [run for run in (campaign / "runs").iterdir() if run.is_dir()]
    checks["task_support_present"] = all(
        (run / "task-support").is_dir() for run in run_dirs
    )
    if not checks["task_support_present"]:
        errors.append("one or more run task-support trees are missing")
    schedule = json.loads((campaign / "schedule.json").read_text(encoding="utf-8"))
    checks["scheduled_runs"] = len(schedule)
    checks["pairing_manifest_present"] = (campaign / V3_PAIRING_FILE).is_file()
    if not checks["pairing_manifest_present"]:
        errors.append("paired-prefix manifest is missing")
    checks["framework_id"] = framework.framework_key
    checks["task_id"] = task.task_id
    from .evaluator import shared_local_evaluator_root, task_local_evaluator_root

    checks["evaluator_scheduler_paths"] = {
        "task": str(task_local_evaluator_root(task.task_id)),
        "host": str(shared_local_evaluator_root()),
    }
    support_hashes = {
        tree_hash(run / "task-support")
        for run in run_dirs
        if (run / "task-support").is_dir()
    }
    checks["task_support_hashes"] = sorted(support_hashes)
    if len(support_hashes) != 1:
        errors.append("v3 run task-support trees are not byte-identical")
    manifest = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    expected_hashes = {
        "protocol_hash": spec.protocol_hash,
        "task_hash": sha256_json(task_hash_payload(task)),
        "framework_hash": sha256_json(framework_hash_payload(framework)),
    }
    checks["input_hashes"] = expected_hashes
    for name, expected in expected_hashes.items():
        if manifest.get(name) != expected:
            errors.append(f"campaign {name} does not match its loaded input")
    if repo_root is None:
        repo_root = next(
            (
                root
                for root in (Path.cwd(), *campaign.parents)
                if (root / "experiments/c0c3_factorial").is_dir()
            ),
            Path.cwd(),
        )
    checks["runtime_hashes"] = {
        "at_creation": manifest.get("scientific_runtime_hash"),
        "current": scientific_runtime_hash(
            repo_root,
            task=task,
            framework=framework,
        ),
    }
    checks["clock_timezone"] = {
        "utc_now": utc_now(),
        "timezone_env": os.environ.get("TZ"),
        "platform_timezone": list(__import__("time").tzname),
    }
    checks["codex_cli"] = {
        "path": shutil.which("codex"),
        "non_scientific_probe_required_before_official_launch": True,
    }
    if checks["codex_cli"]["path"] is None:
        errors.append("Codex CLI is not available on PATH")
    checks["modal"] = {
        "profile_config_present": (Path.home() / ".modal.toml").is_file(),
        "required_for_backend": task.preferred_backend.value
        in {
            "modal",
            "hybrid_modal",
        },
    }
    if (
        checks["modal"]["required_for_backend"]
        and not checks["modal"]["profile_config_present"]
    ):
        errors.append("Modal backend selected but no Modal profile config was found")
    lock_receipts = []
    for lock in campaign.rglob("*.lock"):
        try:
            content = lock.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            content = "[unreadable]"
        lock_receipts.append(
            {"path": lock.relative_to(campaign).as_posix(), "content": content[:500]}
        )
    checks["lock_receipts"] = lock_receipts
    if sys.platform == "darwin":
        try:
            power = subprocess.run(
                ("pmset", "-g", "batt"),
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            checks["power"] = {
                "returncode": power.returncode,
                "summary": power.stdout.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            checks["power"] = {"error": str(error)}
    try:
        import torch

        tensor = torch.tensor([1.0, 2.0])
        checks["accelerator"] = {
            "torch_version": torch.__version__,
            "cpu_tensor_operation": float((tensor * 2).sum().item()),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except (ImportError, RuntimeError) as error:
        checks["accelerator"] = {"available": False, "error": str(error)}
        # The unified protocol also supports command-only tasks whose runtime
        # has no tensor dependency.  Record accelerator availability here, but
        # leave task-specific dependency enforcement to calibration/validation.
    checks["recovery_smoke_test"] = (
        "covered_by tests/test_c0c3_v3.py before official launch"
    )
    return {
        "schema_version": "3.0",
        "checked_at": utc_now(),
        "valid": not errors,
        "checks": checks,
        "errors": errors,
    }
