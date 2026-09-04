"""Deterministic v3 provenance, manipulation packets, and campaign audit."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import math
from collections import Counter
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import candidate_hash
from .spec import FactorialSpec, TaskSpec
from .state import SearchController, append_jsonl, atomic_json, utc_now


def _architecture_ir(path: Path) -> dict[str, int]:
    if path.suffix != ".py":
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {"parse_error": 1}
    counts = Counter(type(node).__name__ for node in ast.walk(tree))
    return dict(sorted(counts.items()))


def candidate_delta(
    *, parent: Path, child: Path, editable_paths: tuple[str, ...]
) -> dict[str, Any]:
    changed_files = 0
    additions = 0
    deletions = 0
    diff_parts = []
    architecture = {}
    for relative in editable_paths:
        before = (parent / relative).read_text(encoding="utf-8", errors="replace")
        after = (child / relative).read_text(encoding="utf-8", errors="replace")
        lines = list(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"parent/{relative}",
                tofile=f"candidate/{relative}",
                lineterm="",
            )
        )
        if lines:
            changed_files += 1
            additions += sum(
                line.startswith("+") and not line.startswith("+++") for line in lines
            )
            deletions += sum(
                line.startswith("-") and not line.startswith("---") for line in lines
            )
            diff_parts.extend(lines)
        architecture[relative] = {
            "parent": _architecture_ir(parent / relative),
            "candidate": _architecture_ir(child / relative),
        }
    diff_text = "\n".join(diff_parts) + ("\n" if diff_parts else "")
    normalized_delta = "\n".join(
        line.strip()
        for line in diff_parts
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )
    return {
        "schema_version": "3.0",
        "changed_files": changed_files,
        "added_lines": additions,
        "deleted_lines": deletions,
        "diff_sha256": hashlib.sha256(diff_text.encode()).hexdigest(),
        "semantic_delta_fingerprint": hashlib.sha256(
            normalized_delta.encode()
        ).hexdigest(),
        "architecture_ir": architecture,
        "diff": diff_text,
    }


def record_candidate_provenance(
    *,
    run_dir: Path,
    task: TaskSpec,
    opportunity: int,
    parent_id: str,
    candidate_id: str,
    proposal_type: str,
    hypothesis: str,
    intended_edit: str,
    mechanism: str,
    evidence: str,
) -> dict[str, Any]:
    parent = run_dir / "candidates" / parent_id
    child = run_dir / "candidates" / candidate_id
    delta = candidate_delta(
        parent=parent, child=child, editable_paths=task.editable_paths
    )
    provenance = {
        **delta,
        "recorded_at": utc_now(),
        "parent_id": parent_id,
        "candidate_id": candidate_id,
        "proposal_type": proposal_type,
        "agent_claims": {
            "hypothesis": hypothesis,
            "intended_edit": intended_edit,
            "mechanism": mechanism,
            "evidence": evidence,
        },
    }
    root = run_dir / "opportunities" / f"{opportunity:04d}"
    atomic_json(root / "candidate-provenance.json", provenance)
    return provenance


def write_manipulation_packet(
    *,
    campaign: Path,
    run_dir: Path,
    opportunity: int,
    candidate_id: str,
    proposal_type: str,
    provenance: dict[str, Any],
) -> Path:
    """Export a blinded source-authoritative check at matched checkpoints."""

    packet_id = hashlib.sha256(
        f"{run_dir.name}:{opportunity}:{candidate_id}".encode()
    ).hexdigest()[:24]
    output = campaign / "review" / "v3-manipulation" / "packets"
    output.mkdir(parents=True, exist_ok=True)
    packet = {
        "schema_version": "3.0",
        "packet_id": packet_id,
        "source_delta": provenance["diff"],
        "architecture_ir": provenance["architecture_ir"],
        "agent_claims": provenance["agent_claims"],
        "annotation": {
            "old_assumption_identifiable": None,
            "new_mechanism_implemented": None,
            "distinct_from_recent_lineage": None,
            "primarily_tuning_pruning_or_deletion": None,
            "cleanly_attributable": None,
            "feasible_under_task_contract": None,
            "novelty_score": None,
            "reviewer_notes": "",
        },
    }
    path = output / f"{packet_id}.json"
    atomic_json(path, packet)
    mapping = {
        "schema_version": "3.0",
        "packet_id": packet_id,
        "run_id": run_dir.name,
        "opportunity": opportunity,
        "proposal_type": proposal_type,
        "candidate_id": candidate_id,
    }
    append_jsonl(
        campaign / "review" / "v3-manipulation" / "private-mapping.jsonl",
        mapping,
    )
    return path


def audit_campaign(
    campaign: Path, *, spec: FactorialSpec, task: TaskSpec
) -> dict[str, Any]:
    """Check state/events/snapshots without mutating scientific artifacts."""

    errors = []
    warnings = []
    seen_sessions: dict[str, str] = {}
    run_reports = []
    for run_dir in sorted((campaign / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        try:
            controller = SearchController.load(run_dir, spec)
        except (OSError, ValueError) as error:
            errors.append(f"{run_dir.name}: invalid state: {error}")
            continue
        completed_by_opportunity: Counter[int] = Counter()
        active_by_opportunity: Counter[int] = Counter()
        if (run_dir / "events.jsonl").is_file():
            for line in (
                (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"{run_dir.name}: malformed event JSON")
                    continue
                opportunity = record.get("opportunity")
                if record.get("event") == "proposal_completed" and isinstance(
                    opportunity, int
                ):
                    completed_by_opportunity[opportunity] += 1
                if record.get("event") == "proposal_started" and isinstance(
                    opportunity, int
                ):
                    active_by_opportunity[opportunity] += 1
                session = record.get("conversation_session_id")
                if isinstance(session, str):
                    owner = seen_sessions.setdefault(session, run_dir.name)
                    if owner != run_dir.name:
                        errors.append(
                            f"conversation session reused by {owner} and {run_dir.name}"
                        )
        duplicate = [
            key for key, value in completed_by_opportunity.items() if value != 1
        ]
        if duplicate:
            errors.append(f"{run_dir.name}: duplicate completion records {duplicate}")
        expected = set(range(1, controller.state.proposals_used + 1))
        if set(completed_by_opportunity) != expected:
            errors.append(
                f"{run_dir.name}: completed event sequence does not match state"
            )
        for identifier in controller.state.candidates:
            snapshot = run_dir / "candidates" / identifier
            if (
                snapshot.is_dir()
                and candidate_hash(snapshot, task.editable_paths) != identifier
            ):
                errors.append(f"{run_dir.name}: candidate hash mismatch {identifier}")
        unmatched_dispatches = []
        for dispatch in run_dir.glob("opportunities/*/remote-dispatch.json"):
            payload = json.loads(dispatch.read_text(encoding="utf-8"))
            if payload.get("status") not in {
                "completed_result_received",
                "client_transport_failed",
            }:
                unmatched_dispatches.append(str(dispatch.relative_to(run_dir)))
        if unmatched_dispatches:
            warnings.append(
                f"{run_dir.name}: remote dispatches need retrieval: "
                f"{unmatched_dispatches}"
            )
        run_reports.append(
            {
                "run_id": run_dir.name,
                "proposals": controller.state.proposals_used,
                "active": (
                    controller.state.active.index
                    if controller.state.active is not None
                    else None
                ),
                "candidate_count": len(controller.state.candidates),
            }
        )
    pairing_path = campaign / "paired-prefix.json"
    if pairing_path.is_file():
        pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
        for pair in pairing.get("pairs", []):
            leader = SearchController.load(
                campaign / "runs" / pair["leader_run_id"], spec
            ).state
            shadow = SearchController.load(
                campaign / "runs" / pair["shadow_run_id"], spec
            ).state
            fork = int(pair["fork_opportunity"])
            if max(leader.next_opportunity, shadow.next_opportunity) <= fork:
                left = leader.to_dict()
                right = shadow.to_dict()
                for value in (left, right):
                    for key in ("run_id", "condition", "revision"):
                        value.pop(key, None)
                if left != right:
                    errors.append(
                        f"paired prefix diverged for {pair['leader_run_id']} and "
                        f"{pair['shadow_run_id']}"
                    )
    return {
        "schema_version": "3.0",
        "audited_at": utc_now(),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "runs": run_reports,
    }


def summarize_campaign(campaign: Path, *, spec: FactorialSpec) -> dict[str, Any]:
    """Build trajectory outcomes and charge shared prefixes exactly once."""

    pricing = json.loads(
        (campaign / "inputs/v3-pricing.json").read_text(encoding="utf-8")
    )
    framework = json.loads(
        (campaign / "inputs/framework.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (campaign / "inputs/protocol.json").read_text(encoding="utf-8")
    )
    model_name = str(protocol["model"]["name"])
    rates = dict(pricing.get("models", {}).get(model_name, {}))
    runs = []
    physical = Counter()
    for run_dir in sorted((campaign / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        controller = SearchController.load(run_dir, spec)
        proposals = []
        starts: dict[int, str] = {}
        selection_counts = Counter()
        fingerprints = set()
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") == "proposal_started" and isinstance(
                event.get("opportunity"), int
            ):
                starts[int(event["opportunity"])] = str(event.get("timestamp"))
                continue
            if event.get("event") != "proposal_completed":
                continue
            proposals.append(event)
            for parent in event.get("selected_parent_ids", []):
                selection_counts[str(parent)] += 1
            provenance = (
                run_dir
                / "opportunities"
                / f"{int(event['opportunity']):04d}"
                / "candidate-provenance.json"
            )
            if provenance.is_file():
                value = json.loads(provenance.read_text(encoding="utf-8"))
                fingerprints.add(str(value.get("semantic_delta_fingerprint")))
            usage = event.get("usage_increment", {})
            if not event.get("shared_prefix"):
                for key in (
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "reasoning_output_tokens",
                ):
                    physical[key] += int(usage.get(key, 0) or 0)
        total_selections = sum(selection_counts.values())
        entropy = (
            -sum(
                (count / total_selections) * math.log(count / total_selections)
                for count in selection_counts.values()
            )
            if total_selections
            else 0.0
        )
        usage = controller.state.usage
        pipeline_wall_seconds = 0.0
        queue_seconds = 0.0
        for event in proposals:
            started = starts.get(int(event["opportunity"]))
            if started:
                with suppress(ValueError):
                    pipeline_wall_seconds += (
                        datetime.fromisoformat(str(event["timestamp"]))
                        - datetime.fromisoformat(started)
                    ).total_seconds()
            queue = (
                run_dir
                / "opportunities"
                / f"{int(event['opportunity']):04d}"
                / "evaluator-queue.json"
            )
            if queue.is_file():
                value = json.loads(queue.read_text(encoding="utf-8"))
                queue_seconds += float(value.get("wait_seconds", 0.0) or 0.0)
        cost = None
        required_rates = {
            "input_per_million",
            "cached_input_per_million",
            "output_per_million",
        }
        if required_rates <= rates.keys():
            uncached = max(0, usage.input_tokens - usage.cached_input_tokens)
            cost = (
                uncached * float(rates["input_per_million"])
                + usage.cached_input_tokens * float(rates["cached_input_per_million"])
                + usage.output_tokens * float(rates["output_per_million"])
            ) / 1_000_000
        runs.append(
            {
                "run_id": run_dir.name,
                "condition": controller.state.condition,
                "proposals": len(proposals),
                "valid": sum(
                    bool(row.get("evaluation", {}).get("valid")) for row in proposals
                ),
                "retained": sum(bool(row.get("retained")) for row in proposals),
                "assumption_opportunities": sum(
                    row.get("proposal_type") == "assumption_changing"
                    for row in proposals
                ),
                "semantic_delta_fingerprints": len(fingerprints),
                "parent_selection_entropy": entropy,
                "pipeline_wall_seconds": pipeline_wall_seconds,
                "evaluator_queue_seconds": queue_seconds,
                "evaluator_runtime_seconds": controller.state.evaluator_seconds_used,
                "best_fitness": controller.state.candidates[
                    controller.state.incumbent_id
                ].fitness,
                "usage": controller.state.usage.__dict__,
                "list_price_equivalent": cost,
                "unpriced": cost is None,
            }
        )
    return {
        "schema_version": "3.0",
        "generated_at": utc_now(),
        "framework_id": framework["framework_id"],
        "model": model_name,
        "runs": runs,
        "physical_usage_charge_shared_prefix_once": dict(physical),
        "notes": {
            "experimental_unit": "trajectory",
            "shared_prefix": (
                "present in both trajectories; physical usage counted once"
            ),
            "semantic_fingerprints": (
                "diagnostic only; blinded human review remains authoritative"
            ),
        },
    }


def build_release_manifest(campaign: Path, *, output: Path) -> Path:
    """Write a secrets-aware content manifest without changing campaign files."""

    excluded_parts = {"private", ".git"}
    excluded_names = {
        ".campaign-runner.lock",
        ".stage-gate.lock",
        ".conversation-session-registry.lock",
    }
    files = []
    for path in sorted(campaign.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.name in excluded_names
            or any(part in excluded_parts for part in path.relative_to(campaign).parts)
        ):
            continue
        files.append(
            {
                "path": path.relative_to(campaign).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    atomic_json(
        output,
        {
            "schema_version": "3.0",
            "generated_at": utc_now(),
            "campaign": campaign.name,
            "files": files,
            "excluded": ["private review mappings", "lock files", "git metadata"],
        },
    )
    return output
