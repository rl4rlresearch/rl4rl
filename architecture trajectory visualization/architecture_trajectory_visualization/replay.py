"""Read-only, source-backed replay of recorded architecture search trajectories.

The module never imports or executes candidate programs. Events identify proposal
occurrences; candidate IDs identify snapshots, and neither is a Git commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import subprocess
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from architecture_trajectory_visualization.paths import REPO_ROOT

SCHEMA_VERSION = "architecture-replay/1"
MAX_SAFE_INTEGER = 2**53 - 1
ARCHIVES = {
    "tiny-v21": "rl4rl-tiny-adderboard-through-100-c0-c3-20260908.zip",
    "uci-har-pareto-v21": "rl4rl-uci-har-c0-c3-20260908.zip",
}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_safe(value: Any) -> Any:
    """Keep large integer measurements exact when consumed by JavaScript."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and abs(value) > MAX_SAFE_INTEGER:
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def source_revision(repo_root: Path | str) -> str:
    provenance = dataset_provenance(repo_root)
    if isinstance(provenance.get("source_revision"), str):
        return provenance["source_revision"]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def dataset_provenance(repo_root: Path | str) -> dict[str, Any]:
    return _dict(_read_json(Path(repo_root) / "architecture-replay-source.json"))


def _scope(scope: str) -> None:
    if scope not in {"dashboard", "archive"}:
        raise ValueError("scope must be dashboard or archive")


def _dashboard_policy(run_id: str, scope: str) -> tuple[bool, int | None]:
    _scope(scope)
    if scope == "archive":
        return True, None
    # Import only when called, so the HTTP dashboard can import this module.
    from experiments.live_trajectory_dashboard import (
        dashboard_proposal_cap,
        dashboard_run_visible,
    )

    return dashboard_run_visible(run_id), dashboard_proposal_cap(run_id)


def _campaign_path(repo_root: Path | str, campaign: Path | str) -> Path:
    repo_root = Path(repo_root).resolve()
    root = (repo_root / "data" / "c0c3").resolve()
    candidate = Path(campaign)
    if not candidate.is_absolute():
        candidate = (
            repo_root / candidate if len(candidate.parts) > 1 else root / candidate
        )
    candidate = candidate.resolve()
    if candidate.parent != root:
        raise ValueError("Campaign must be a direct child of data/c0c3")
    return candidate


def _run_path(repo_root: Path | str, campaign: Path | str, run_id: str) -> Path:
    if not SAFE_NAME.fullmatch(run_id):
        raise ValueError("Invalid run ID")
    runs = (_campaign_path(repo_root, campaign) / "runs").resolve()
    run = (runs / run_id).resolve()
    if run.parent != runs or not run.is_dir():
        raise FileNotFoundError("Unknown run")
    return run


def _events(path: Path, diagnostics: list[str]) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                result.append(item)
        except ValueError:
            diagnostics.append(f"Unreadable events.jsonl line {number}")
    return result


def metric_definitions(campaign: Path | str) -> list[dict[str, Any]]:
    campaign = Path(campaign)
    task = _dict(_read_json(campaign / "inputs" / "task.json"))
    task_id = str(task.get("task_id", campaign.name)).lower()
    extension = _dict(task.get("extension_options"))
    objective = extension.get("dashboard_objective", task.get("objective_metric"))
    direction = task.get("objective_direction", "minimize")
    if "nanogpt" in task_id:
        defaults = [
            ("val_bpb", "Validation bits / byte", "minimize", "bpb", 4),
            ("num_params_M", "Parameters", "minimize", "million", 2),
            ("depth", "Depth", None, "layers", 0),
            ("peak_vram_mb", "Peak VRAM", "minimize", "MB", 1),
        ]
    elif "har" in task_id:
        defaults = [
            ("validation_accuracy", "Validation accuracy", "maximize", "ratio", 4),
            ("inference_macs", "Inference MACs", "minimize", "MACs", 0),
            ("parameters", "Parameters", "minimize", "parameters", 0),
        ]
    elif "kws" in task_id:
        defaults = [
            ("validation_accuracy", "Validation accuracy", "maximize", "ratio", 4),
            ("inference_cost", "Inference cost", "minimize", "cost", 0),
            ("mean_recurrent_steps", "Mean recurrent steps", "minimize", "steps", 2),
            ("parameters", "Parameters", "minimize", "parameters", 0),
        ]
    elif "fashion" in task_id:
        defaults = [
            ("validation_accuracy", "Validation accuracy", "maximize", "ratio", 4),
            (
                "validation_cross_entropy",
                "Validation cross entropy",
                "minimize",
                "loss",
                4,
            ),
            ("parameters", "Parameters", "minimize", "parameters", 0),
        ]
    else:
        defaults = [
            ("parameters", "Parameters", "minimize", "parameters", 0),
            ("accuracy", "Accuracy", "maximize", "ratio", 4),
        ]
    objective = objective or defaults[0][0]
    definitions = [
        {
            "key": k,
            "label": label,
            "direction": d,
            "unit": unit,
            "precision": precision,
            "is_objective": k == objective,
        }
        for k, label, d, unit, precision in defaults
    ]
    if objective not in {row["key"] for row in definitions}:
        definitions.insert(
            0,
            {
                "key": objective,
                "label": str(objective).replace("_", " ").title(),
                "direction": direction,
                "unit": "",
                "precision": 4,
                "is_objective": True,
            },
        )
    for row in definitions:
        if row["is_objective"] and task.get("objective_direction"):
            row["direction"] = direction
    known = {row["key"] for row in definitions}
    for key in task.get("public_feedback_metrics", []):
        if isinstance(key, str) and key not in known:
            definitions.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "direction": None,
                    "unit": "",
                    "precision": 4,
                    "is_objective": False,
                }
            )
            known.add(key)
    return definitions


def metric_contract(campaign: Path | str) -> dict[str, Any]:
    """Preserve the recorded scientific comparison contract without rerunning it."""
    task = _dict(_read_json(Path(campaign) / "inputs" / "task.json"))
    fields = (
        "task_id",
        "objective_metric",
        "objective_direction",
        "qualification_metric",
        "qualification_minimum",
        "preferred_backend",
        "public_feedback_metrics",
    )
    return {key: task.get(key) for key in fields if key in task}


def resolve_candidate_source(
    repo_root: Path | str,
    campaign: Path | str,
    run_id: str,
    candidate_id: str | None,
    *,
    artifact_path: str | None = None,
    proposal: int | None = None,
    parent_ids: list[str] | None = None,
    failure_kind: str | None = None,
) -> dict[str, Any]:
    """Return exact saved Python source, with bounded reads and no execution."""
    run = _run_path(repo_root, campaign, run_id)
    result: dict[str, Any] = {
        "candidate_id": candidate_id,
        "available": False,
        "availability": "missing",
        "source_hash": None,
        "files": [],
        "sources": {},
        "diagnostics": [],
        "ir": None,
    }
    if not candidate_id:
        result["diagnostics"].append("No candidate was recorded for this occurrence")
        return result
    if not isinstance(candidate_id, str) or not SAFE_NAME.fullmatch(candidate_id):
        raise ValueError("Invalid candidate ID")
    candidates = (run / "candidates").resolve()
    expected = (candidates / candidate_id).resolve()
    if not candidates.is_relative_to(run) or expected.parent != candidates:
        raise ValueError("Candidate snapshot escapes the run")
    if artifact_path:
        recorded = (run / artifact_path.replace("\\", "/")).resolve()
        if recorded.parent != candidates or not SAFE_NAME.fullmatch(recorded.name):
            result["diagnostics"].append(
                "Recorded artifact path escapes the candidate snapshot collection"
            )
            result["availability"] = "invalid_path"
            return result
        if recorded != expected:
            if failure_kind == "infrastructure_interruption":
                from experiments.ontology_categorical_inventory import (
                    _interrupted_workspace_evidence,
                )

                evidence = (
                    _interrupted_workspace_evidence(
                        run,
                        {"opportunity": proposal, "parent_ids": parent_ids or []},
                        recorded.name,
                    )
                    if type(proposal) is int and proposal > 0
                    else None
                )
                if evidence is None:
                    result.update(
                        availability="unresolved_recovery_fallback",
                        identity_relation="unresolved_recovery_fallback",
                        recorded_artifact_path=artifact_path,
                    )
                    result["diagnostics"].append(
                        "Interrupted-opportunity parent reference does not identify "
                        "attempted source: an exact editable-file workspace match "
                        "was not established"
                    )
                    return result
                result["interruption_evidence"] = evidence
            result["identity_relation"] = "recorded_alias"
            result["diagnostics"].append(
                "Recorded artifact references another snapshot; candidate identity "
                "and source identity are preserved separately"
            )
            expected = recorded
    result.setdefault("identity_relation", "direct")
    result["source_candidate_id"] = expected.name
    if not expected.is_dir():
        result["diagnostics"].append("Saved candidate source is absent")
        return result
    identity_diagnostic_count = len(result["diagnostics"])
    paths = sorted(expected.rglob("*.py"))
    if len(paths) > 30:
        result["diagnostics"].append("Source inventory exceeds the 30-file limit")
    for path in paths[:30]:
        if not path.resolve().is_relative_to(expected) or path.is_symlink():
            result["diagnostics"].append("Skipped source symlink or escaping path")
            continue
        if path.stat().st_size > 500_000:
            result["diagnostics"].append(f"Source file exceeds 500 KB: {path.name}")
            continue
        try:
            raw = path.read_bytes()
            content = raw.decode("utf-8")
        except (OSError, UnicodeError):
            result["diagnostics"].append(f"Unreadable source file: {path.name}")
            continue
        name = path.relative_to(expected).as_posix()
        result["sources"][name] = content
        result["files"].append(
            {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        )
    for name in ("architecture.json", "arch.json", "architecture_ir.json"):
        path = expected / name
        if path.is_file() and not path.is_symlink() and path.stat().st_size < 500_000:
            ir = _dict(_read_json(path))
            graph = _dict(ir.get("graph", ir))
            if isinstance(graph.get("nodes"), list):
                result["ir"] = ir
                raw = path.read_bytes()
                result["files"].append(
                    {
                        "path": name,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                    }
                )
                break
    result["available"] = bool(result["sources"] or result["ir"])
    result["availability"] = (
        (
            "partial"
            if len(result["diagnostics"]) > identity_diagnostic_count
            else "available"
        )
        if result["available"]
        else "missing"
    )
    result["source_hash"] = _hash(result["files"]) if result["files"] else None
    result["artifact_path"] = expected.relative_to(Path(repo_root).resolve()).as_posix()
    return result


def _source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if key not in {"sources", "ir"}}


def _evaluation_rows(
    result: dict[str, Any], occurrence_id: str
) -> tuple[list[dict[str, Any]], str | None]:
    rows = []
    recorded_evaluations = result.get("evaluations")
    for index, item in enumerate(
        recorded_evaluations if isinstance(recorded_evaluations, list) else []
    ):
        if isinstance(item, dict):
            rows.append(
                {**item, "id": item.get("id", f"{occurrence_id}:evaluation:{index}")}
            )
    decision = result.get("decision_evaluation_id")
    embedded = _dict(result.get("evaluation"))
    if embedded:
        decision = f"{occurrence_id}:decision"
        rows.append({**embedded, "id": decision, "role": "recorded_decision"})
    if decision not in {row["id"] for row in rows}:
        decision = rows[0]["id"] if len(rows) == 1 else None
    return rows, decision


def load_run_replay(
    repo_root: Path | str,
    campaign: Path | str,
    run_id: str,
    *,
    scope: str = "dashboard",
) -> dict[str, Any]:
    """Normalize a run without deleting failed, repeated, or missing-source attempts."""
    visible, cap = _dashboard_policy(run_id, scope)
    if not visible:
        raise FileNotFoundError(
            "Run is excluded from dashboard scope; use archive scope"
        )
    root = _campaign_path(repo_root, campaign)
    run = _run_path(repo_root, root, run_id)
    diagnostics: list[str] = []
    events = _events(run / "events.jsonl", diagnostics)
    state = _dict(_read_json(run / "state.json"))
    manifest = _dict(_read_json(run / "manifest.json"))
    assignment = _dict(manifest.get("assignment"))
    completed: dict[int, dict[str, Any]] = {}
    started: dict[int, dict[str, Any]] = {}
    for event in events:
        proposal = event.get("opportunity")
        if not isinstance(proposal, int) or isinstance(proposal, bool):
            continue
        if event.get("event") == "proposal_started":
            started[proposal] = event
        elif event.get("event") == "proposal_completed":
            if proposal in completed and completed[proposal] != event:
                diagnostics.append(
                    f"Multiple completion records for proposal {proposal}; "
                    "latest record used"
                )
            completed[proposal] = event
    numbers = set(started) | set(completed)
    opportunity_root = run / "opportunities"
    if opportunity_root.is_dir():
        numbers.update(
            int(path.name)
            for path in opportunity_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
    active = _dict(state.get("active"))
    if isinstance(active.get("index"), int):
        numbers.add(active["index"])
    baseline = _dict(manifest.get("baseline"))
    creation = next(
        (event for event in events if event.get("event") == "run_created"), {}
    )
    seed_id = creation.get("seed_candidate_id", baseline.get("candidate_id"))
    if not seed_id:
        seed = next(
            (
                c
                for c in _dict(state.get("candidates")).values()
                if isinstance(c, dict) and c.get("created_opportunity") == 0
            ),
            {},
        )
        seed_id = seed.get("candidate_id")
        baseline = {**seed, **baseline}
    seed_metrics = _dict(baseline.get("metrics"))
    seed_source = resolve_candidate_source(repo_root, root, run_id, seed_id)
    run_key = f"{root.name}/{run_id}"
    seed_occurrence_id = f"{run_key}:0"
    seed_eval = {
        "id": f"{seed_occurrence_id}:baseline",
        "metrics": seed_metrics,
        "fitness": baseline.get("fitness"),
        "role": "recorded_baseline",
    }
    occurrences = [
        {
            "id": seed_occurrence_id,
            "proposal": 0,
            "candidate_id": seed_id,
            "parent_ids": [],
            "parent_occurrence_ids": [],
            "status": "seed",
            "retained": True,
            "valid": None,
            "metrics": seed_metrics,
            "incumbent_after": seed_id,
            "portfolio_after": [seed_id] if seed_id else [],
            "timestamp": creation.get("timestamp"),
            "started_at": None,
            "completed_at": creation.get("timestamp"),
            "retention_decision": "seed",
            "evaluations": [seed_eval],
            "decision_evaluation_id": seed_eval["id"],
            "source": _source_metadata(seed_source),
            "mechanism": "Starting architecture",
        }
    ]
    latest = {seed_id: seed_occurrence_id} if seed_id else {}
    source_cache = {seed_id: seed_source} if seed_id else {}
    fingerprint_inputs: list[Any] = [manifest, events, state, scope]
    for proposal in sorted(n for n in numbers if n > 0 and (cap is None or n <= cap)):
        directory = opportunity_root / f"{proposal:04d}"
        event = completed.get(proposal, {})
        file_result = _dict(_read_json(directory / "result.json"))
        # Same precedence as run_transcript_payload in the existing dashboard.
        result = {**event, **file_result}
        fingerprint_inputs.append(file_result)
        start = started.get(proposal, {})
        provenance = _dict(_read_json(directory / "candidate-provenance.json"))
        claims = _dict(provenance.get("agent_claims"))
        candidate_id = result.get("candidate_id")
        occurrence_id = f"{run_key}:{proposal}"
        evaluations, decision = _evaluation_rows(result, occurrence_id)
        evaluation = next((e for e in evaluations if e["id"] == decision), {})
        metrics = _dict(evaluation.get("metrics", result.get("metrics")))
        parent_ids = result.get("parent_ids", start.get("selected_parent_ids", []))
        parent_ids = (
            [p for p in parent_ids if isinstance(p, str)]
            if isinstance(parent_ids, list)
            else []
        )
        unresolved = [p for p in parent_ids if p not in latest]
        if unresolved:
            diagnostics.append(f"Proposal {proposal} has unresolved recorded parents")
        finished = bool(
            event or file_result.get("event") == "proposal_completed" or evaluation
        )
        valid = evaluation.get("valid")
        retained = result.get("retained")
        failure = evaluation.get("failure_kind")
        if not finished:
            status = "incomplete"
        elif valid is False:
            status = "invalid" if failure in {None, "nonqualification"} else "failed"
        elif retained is True:
            status = "retained"
        elif retained is False:
            status = "rejected"
        else:
            status = "unknown"
        artifact_path = result.get("artifact_path")
        cache_key = (
            candidate_id,
            artifact_path,
            proposal if failure == "infrastructure_interruption" else None,
        )
        if cache_key not in source_cache:
            try:
                source_cache[cache_key] = resolve_candidate_source(
                    repo_root,
                    root,
                    run_id,
                    candidate_id,
                    artifact_path=artifact_path,
                    proposal=proposal,
                    parent_ids=parent_ids,
                    failure_kind=failure,
                )
            except ValueError as error:
                source_cache[cache_key] = {
                    "available": False,
                    "availability": "invalid_path",
                    "source_hash": None,
                    "files": [],
                    "diagnostics": [str(error)],
                }
        occurrence = {
            "id": occurrence_id,
            "proposal": proposal,
            "candidate_id": candidate_id,
            "parent_ids": parent_ids,
            "parent_occurrence_ids": [latest[p] for p in parent_ids if p in latest],
            "unresolved_parent_ids": unresolved,
            "selected_parent_ids": result.get(
                "selected_parent_ids", start.get("selected_parent_ids", [])
            ),
            "incumbent_after": result.get("incumbent_after"),
            "portfolio_after": result.get("portfolio_after", []),
            "retained": retained,
            "valid": valid,
            "status": status,
            "failure_kind": failure,
            "metrics": metrics,
            "fitness": evaluation.get("fitness"),
            "retention_decision": result.get("retention_decision"),
            "timestamp": result.get("timestamp", start.get("timestamp")),
            "started_at": start.get("timestamp"),
            "completed_at": result.get("timestamp") if finished else None,
            "evaluations": evaluations,
            "decision_evaluation_id": decision,
            "source": _source_metadata(source_cache[cache_key]),
            "artifact_path": artifact_path,
            "source_record": f"opportunities/{proposal:04d}/result.json"
            if file_result
            else "events.jsonl",
        }
        for key in ("hypothesis", "mechanism", "evidence", "intended_edit"):
            occurrence[key] = result.get(key) or claims.get(key)
        occurrences.append(occurrence)
        if candidate_id:
            latest[candidate_id] = occurrence_id
    revision = source_revision(repo_root)
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "campaign": root.name,
            "run_id": run_id,
            "run_key": run_key,
            "scope": scope,
            "source_revision": revision,
            "dataset_provenance": dataset_provenance(repo_root),
            "revision": _hash(
                [
                    revision,
                    fingerprint_inputs,
                    [row["source"].get("source_hash") for row in occurrences],
                ]
            ),
            "condition": assignment.get("condition", state.get("condition")),
            "replicate": assignment.get("block"),
            "proposal_cap": cap,
            "metric_definitions": metric_definitions(root),
            "metric_contract": {
                **metric_contract(root),
                **{
                    key: manifest[key]
                    for key in ("protocol_hash", "task_hash", "framework_hash")
                    if key in manifest
                },
            },
            "occurrences": occurrences,
            "diagnostics": diagnostics,
            "notes": [
                "The slider advances proposal occurrences, not training steps.",
                "Retention and incumbent changes are separate recorded decisions.",
                "Missing source is shown explicitly; current code never replaces it.",
            ],
        }
    )


def discover_catalog(
    repo_root: Path | str,
    *,
    scope: str = "dashboard",
    campaigns: Mapping[str, Path | str] | None = None,
) -> dict[str, Any]:
    _scope(scope)
    repo_root = Path(repo_root).resolve()
    roots = dict(campaigns or {})
    if campaigns is None:
        data = repo_root / "data" / "c0c3"
        if data.is_dir():
            roots = {p.name: p for p in sorted(data.iterdir()) if (p / "runs").is_dir()}
            if scope == "dashboard":
                roots = {
                    key: path
                    for key, path in roots.items()
                    if "invalid-launch" not in key and "quarantine" not in key
                }
        for key in ARCHIVES:
            roots.setdefault(key, data / key)
    result = []
    for key, path in roots.items():
        campaign = _campaign_path(repo_root, path)
        framework = _dict(_read_json(campaign / "inputs" / "framework.json")).get(
            "framework_id", "unknown"
        )
        task = _dict(_read_json(campaign / "inputs" / "task.json"))
        runs = []
        root = campaign / "runs"
        for run in sorted(root.iterdir()) if root.is_dir() else []:
            if not run.is_dir() or not _dashboard_policy(run.name, scope)[0]:
                continue
            manifest = _dict(_read_json(run / "manifest.json"))
            assignment = _dict(manifest.get("assignment"))
            state = _dict(_read_json(run / "state.json"))
            if not manifest and not state and not (run / "events.jsonl").is_file():
                continue
            cap = _dashboard_policy(run.name, scope)[1]
            proposals = state.get("proposals_used")
            if isinstance(proposals, int) and cap is not None:
                proposals = min(proposals, cap)
            runs.append(
                {
                    "id": run.name,
                    "label": run.name,
                    "condition": assignment.get("condition", state.get("condition")),
                    "framework": framework,
                    "replicate": assignment.get("block"),
                    "proposals": proposals,
                    "status": state.get("status", "recorded"),
                    "proposal_cap": cap,
                }
            )
        archive = ARCHIVES.get(campaign.name)
        result.append(
            {
                "key": key,
                "label": campaign.name.replace("-", " "),
                "path": campaign.relative_to(repo_root).as_posix(),
                "available": bool(runs),
                "runs": runs,
                "task_id": task.get("task_id"),
                "framework": framework,
                "metric_definitions": metric_definitions(campaign),
                "metric_contract": metric_contract(campaign),
                "archive": f"data/{archive}" if archive else None,
                "availability_note": None
                if runs
                else "Campaign is not materialized; prepare its GitHub archive.",
            }
        )
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "scope": scope,
            "source_revision": source_revision(repo_root),
            "dataset_provenance": dataset_provenance(repo_root),
            "campaigns": result,
        }
    )


def export_architecture_bundle(
    repo_root: Path | str,
    campaign: Path | str,
    run_id: str,
    *,
    scope: str = "dashboard",
) -> dict[str, Any]:
    """Create a portable replay with parsed graphs, never executable source files."""
    from architecture_trajectory_visualization.graph import extract_architecture

    run = load_run_replay(repo_root, campaign, run_id, scope=scope)
    snapshots = {}
    graph_cache: dict[str | None, dict[str, Any]] = {}
    for occurrence in run["occurrences"]:
        source = resolve_candidate_source(
            repo_root,
            campaign,
            run_id,
            occurrence.get("candidate_id"),
            artifact_path=occurrence.get("artifact_path"),
            proposal=occurrence.get("proposal"),
            parent_ids=occurrence.get("parent_ids"),
            failure_kind=occurrence.get("failure_kind"),
        )
        source_hash = source.get("source_hash")
        if source_hash not in graph_cache:
            graph_cache[source_hash] = extract_architecture(
                source.get("sources", {}), source_hash=source_hash, ir=source.get("ir")
            )
        snapshots[occurrence["id"]] = {
            **graph_cache[source_hash],
            "occurrence_id": occurrence["id"],
            "candidate_id": occurrence.get("candidate_id"),
            "source_id": source.get("source_candidate_id")
            if source["available"]
            else None,
            "source_hash": source_hash,
            "source_metadata": _source_metadata(source),
            "source_revision": run["source_revision"],
            "run_revision": run["revision"],
        }
    bundle = {
        "schema_version": "architecture-bundle/1",
        "run": run,
        "snapshots": snapshots,
    }
    validate_architecture_bundle(bundle)
    return _json_safe(bundle)


def validate_architecture_bundle(bundle: Any) -> None:
    """Validate portable structure and graph references, not scientific authenticity."""
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema_version") != "architecture-bundle/1"
    ):
        raise ValueError("Unsupported architecture bundle schema")
    run = _dict(bundle.get("run"))
    snapshots = bundle.get("snapshots")
    occurrences = run.get("occurrences")
    if run.get("schema_version") != SCHEMA_VERSION or not isinstance(occurrences, list):
        raise ValueError("Bundle must contain a normalized replay run")
    if not isinstance(snapshots, dict) or not occurrences:
        raise ValueError("Bundle must contain occurrence snapshots")
    ids = [row.get("id") for row in occurrences if isinstance(row, dict)]
    if (
        len(ids) != len(occurrences)
        or not all(isinstance(key, str) for key in ids)
        or len(set(ids)) != len(ids)
        or set(ids) != set(snapshots)
    ):
        raise ValueError("Bundle occurrence IDs and snapshot keys must match uniquely")
    for occurrence in occurrences:
        snapshot = snapshots[occurrence["id"]]
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("occurrence_id") != occurrence["id"]
            or snapshot.get("candidate_id") != occurrence.get("candidate_id")
        ):
            raise ValueError("Snapshot occurrence identity mismatch")
        nodes, edges = snapshot.get("nodes"), snapshot.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("Snapshot must include nodes and edges")
        node_ids = [node.get("id") for node in nodes if isinstance(node, dict)]
        if (
            len(node_ids) != len(nodes)
            or not all(isinstance(key, str) for key in node_ids)
            or len(set(node_ids)) != len(node_ids)
        ):
            raise ValueError("Snapshot node IDs must be unique strings")
        if any(
            not isinstance(edge, dict)
            or edge.get("source") not in node_ids
            or edge.get("target") not in node_ids
            for edge in edges
        ):
            raise ValueError("Snapshot edge references an unknown node")
        metadata = _dict(snapshot.get("source_metadata"))
        expected = _dict(occurrence.get("source")).get("source_hash")
        if (
            metadata.get("source_hash") != expected
            or snapshot.get("source_hash") != expected
        ):
            raise ValueError("Snapshot source hash differs from its replay occurrence")


def lfs_pointer(path: Path | str) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1024:
        return None
    try:
        content = path.read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return None
    if not content.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    oid = re.search(r"^oid sha256:([0-9a-f]{64})$", content, re.MULTILINE)
    size = re.search(r"^size (\d+)$", content, re.MULTILINE)
    return {
        "sha256": oid.group(1) if oid else None,
        "bytes": int(size.group(1)) if size else None,
    }


def verify_archive(
    path: Path | str, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    path = Path(path)
    if lfs_pointer(path):
        raise ValueError(
            f"Git LFS pointer, not ZIP data: {path.name}. "
            f"Run git lfs pull --include='data/{path.name}' in the source checkout."
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if expected_sha256 and actual != expected_sha256:
        raise ValueError("Archive SHA-256 does not match the recorded GitHub archive")
    if not zipfile.is_zipfile(path):
        raise ValueError("Archive is not a ZIP file")
    return {
        "sha256": actual,
        "bytes": path.stat().st_size,
        "checksum_verified": bool(expected_sha256),
    }


def prepare_archive(
    archive: Path | str,
    destination: Path | str,
    *,
    expected_sha256: str | None = None,
    metadata_only: bool = True,
) -> dict[str, Any]:
    """Verify then extract safely. Existing different files are never overwritten."""
    report = verify_archive(archive, expected_sha256=expected_sha256)
    destination = Path(destination).resolve()
    allowed = {".json", ".jsonl", ".py", ".toml", ".md", ".csv", ".tsv"}
    selected: list[tuple[zipfile.ZipInfo, Path]] = []
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        if sum(item.file_size for item in bundle.infolist()) > 12 * 1024**3:
            raise ValueError("Archive uncompressed size exceeds 12 GB")
        for item in bundle.infolist():
            name = item.filename.replace("\\", "/")
            parts = PurePosixPath(name)
            if (
                parts.is_absolute()
                or ".." in parts.parts
                or re.match(r"^[A-Za-z]:", name)
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError("Unsafe archive member path or symbolic link")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or target == destination:
                if item.is_dir() and target == destination:
                    continue
                raise ValueError("Archive member escapes destination")
            if item.is_dir() or (
                metadata_only and target.suffix.lower() not in allowed
            ):
                continue
            if target.as_posix().casefold() in seen:
                raise ValueError("Duplicate archive member path")
            seen.add(target.as_posix().casefold())
            if target.exists():
                if (
                    not target.is_file()
                    or hashlib.sha256(target.read_bytes()).digest()
                    != hashlib.sha256(bundle.read(item)).digest()
                ):
                    raise ValueError(f"Refusing to overwrite existing artifact: {name}")
                continue
            selected.append((item, target))
        # Complete validation before creating output paths. ZIP reads verify CRC.
        for item, target in selected:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output, bundle.open(item) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
    return {
        **report,
        "files_extracted": len(selected),
        "destination": str(destination),
        "metadata_only": metadata_only,
    }


def _archive_expected_hash(repo_root: Path, archive: Path) -> str | None:
    transfer = _dict(
        _read_json(repo_root / "data" / "rl4rl-tiny-har-transfer-20260908.json")
    )
    for row in transfer.get("archives", []):
        if isinstance(row, dict) and row.get("archive") == archive.name:
            return row.get("sha256")
    try:
        relative = archive.resolve().relative_to(repo_root.resolve()).as_posix()
        content = subprocess.check_output(
            ["git", "-C", str(repo_root), "show", f"HEAD:{relative}"],
            stderr=subprocess.DEVNULL,
            timeout=3,
            text=True,
        )
        match = re.search(r"^oid sha256:([0-9a-f]{64})$", content, re.MULTILINE)
        return match.group(1) if match else None
    except (ValueError, OSError, subprocess.SubprocessError, UnicodeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=REPO_ROOT
    )
    sub = parser.add_subparsers(dest="command", required=True)
    catalog = sub.add_parser("catalog")
    catalog.add_argument(
        "--scope", choices=["dashboard", "archive"], default="dashboard"
    )
    export = sub.add_parser("export")
    export.add_argument("--campaign", required=True)
    export.add_argument("--run", required=True)
    export.add_argument(
        "--scope", choices=["dashboard", "archive"], default="dashboard"
    )
    export.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare-archive")
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("--destination", type=Path, required=True)
    prepare.add_argument("--sha256")
    prepare.add_argument("--all-files", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            payload = discover_catalog(args.repo, scope=args.scope)
        elif args.command == "export":
            payload = export_architecture_bundle(
                args.repo, args.campaign, args.run, scope=args.scope
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(
                f"Exported {len(payload['run']['occurrences'])} snapshots "
                f"to {args.output}"
            )
            return 0
        else:
            expected = args.sha256 or _archive_expected_hash(args.repo, args.archive)
            if not expected:
                raise ValueError("A recorded archive SHA-256 or --sha256 is required")
            payload = prepare_archive(
                args.archive,
                args.destination,
                expected_sha256=expected,
                metadata_only=not args.all_files,
            )
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(2, f"architecture-replay: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
