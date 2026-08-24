#!/usr/bin/env python3
"""Build the trajectory dashboard data and the complete Markdown run report."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


RUN_PATTERN = re.compile(
    r"^(?P<method>ar|oe)-size-h(?P<horizon>8|12|20)-.*-rd(?P<condition>[0-3])-s1(?:-r(?P<retry>\d+))?$"
)
CONDITION_LABELS = {
    "RD0": "Sequential memory · neutral review",
    "RD1": "Sequential memory · assumption challenge",
    "RD2": "Portfolio memory · neutral review",
    "RD3": "Portfolio memory · assumption challenge",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def parse_json_object(value: Any) -> dict[str, Any]:
    """Return a JSON object from either its decoded or serialized form."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def as_bool_metric(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    number = finite_number(value)
    return number is not None and number >= 1.0


def parse_graph(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        last_fence = text.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            text = text[first_newline + 1 : last_fence].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def graph_summary(graph: dict[str, Any] | None) -> tuple[str, str, dict[str, int]]:
    if graph is None:
        return "Unparsed proposal", "No structured Architecture IR was recovered.", {}
    metadata = graph.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    name = metadata.get("name") or metadata.get("architecture") or graph.get("graph_id")
    if not isinstance(name, str) or not name.strip():
        name = "Unnamed architecture"
    mechanism = metadata.get("mechanism_hypothesis")
    if not isinstance(mechanism, str) or not mechanism.strip():
        mechanism = metadata.get("research_next_experiment")
    if not isinstance(mechanism, str) or not mechanism.strip():
        mechanism = "No concise mechanism hypothesis was recorded."
    counts: Counter[str] = Counter()
    nodes = graph.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict) and isinstance(node.get("kind"), str):
                counts[node["kind"]] += 1
    return name.strip(), mechanism.strip(), dict(sorted(counts.items()))


def ar_graph(run_dir: Path, iteration: int) -> dict[str, Any] | None:
    matches = sorted((run_dir / "controller" / "artifacts").glob(f"{iteration:04d}_*.ir.json"))
    if not matches:
        return None
    return load_json(matches[0])


def valid_candidate(metrics: dict[str, Any], *, infrastructure_failure: bool = False) -> bool:
    params = finite_number(metrics.get("parameter_count_metadata"))
    return (
        params is not None
        and params > 0
        and not infrastructure_failure
        and as_bool_metric(metrics.get("execution_ok"))
        and as_bool_metric(metrics.get("transformer_valid"))
        and as_bool_metric(metrics.get("eligible_for_parent"))
    )


def candidate_record(
    *,
    iteration: int,
    candidate_id: str,
    parent_id: str | None,
    architecture_hash: str | None,
    metrics: dict[str, Any],
    graph: dict[str, Any] | None,
    input_tokens: int | float | None,
    output_tokens: int | float | None,
    failure_reason: str,
    infrastructure_failure: bool,
) -> dict[str, Any]:
    params = finite_number(metrics.get("parameter_count_metadata"))
    accuracy = finite_number(metrics.get("public_accuracy"))
    score = finite_number(metrics.get("search_score"))
    name, mechanism, node_kinds = graph_summary(graph)
    is_valid = valid_candidate(metrics, infrastructure_failure=infrastructure_failure)
    return {
        "iteration": iteration,
        "candidateId": candidate_id,
        "parentId": parent_id,
        "architectureHash": architecture_hash,
        "parameterCount": int(params) if params is not None and params.is_integer() else params,
        "publicAccuracy": accuracy,
        "searchScore": score,
        "valid": is_valid,
        "executionOk": as_bool_metric(metrics.get("execution_ok")),
        "transformerValid": as_bool_metric(metrics.get("transformer_valid")),
        "eligibleForParent": as_bool_metric(metrics.get("eligible_for_parent")),
        "infrastructureFailure": infrastructure_failure,
        "failureReason": failure_reason,
        "inputTokens": int(input_tokens or 0),
        "outputTokens": int(output_tokens or 0),
        "architectureName": name,
        "mechanismHypothesis": mechanism,
        "nodeKinds": node_kinds,
    }


def ar_candidates(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_jsonl(run_dir / "controller" / "lineage.jsonl")
    if not rows or rows[0].get("proposal_opportunity") != 0:
        raise ValueError(f"AutoResearch lineage lacks its seed row: {run_dir}")
    seed_row = rows[0]
    seed = candidate_record(
        iteration=0,
        candidate_id=str(seed_row.get("candidate_id", "seed")),
        parent_id=None,
        architecture_hash=str(seed_row.get("code_hash") or seed_row.get("candidate_id") or ""),
        metrics=seed_row,
        graph=load_json(run_dir / "controller" / "accepted_lineage" / "candidate.ir.json"),
        input_tokens=0,
        output_tokens=0,
        failure_reason=str(seed_row.get("failure_stage") or ""),
        infrastructure_failure=bool(seed_row.get("infrastructure_failure")),
    )
    candidates: list[dict[str, Any]] = []
    for row in rows[1:]:
        iteration = int(row["proposal_opportunity"])
        candidates.append(
            candidate_record(
                iteration=iteration,
                candidate_id=str(row.get("candidate_id") or f"proposal-{iteration}"),
                parent_id=str(row["parent_id"]) if row.get("parent_id") else None,
                architecture_hash=str(row.get("code_hash") or row.get("candidate_id") or "") or None,
                metrics=row,
                graph=ar_graph(run_dir, iteration),
                input_tokens=row.get("input_tokens"),
                output_tokens=row.get("output_tokens"),
                failure_reason=str(row.get("failure_stage") or ""),
                infrastructure_failure=bool(row.get("infrastructure_failure")),
            )
        )
    return seed, candidates


def oe_candidates(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_jsonl(run_dir / "controller" / "evolution_trace.jsonl")
    if not rows:
        raise ValueError(f"OpenEvolve trace is empty: {run_dir}")
    first_parent = rows[0].get("parent_metrics")
    if not isinstance(first_parent, dict):
        first_parent = {}
    first_parent_code = parse_graph(rows[0].get("parent_code"))
    seed = candidate_record(
        iteration=0,
        candidate_id=str(rows[0].get("parent_id") or "seed"),
        parent_id=None,
        architecture_hash=(rows[0].get("artifacts") or {}).get("parent_architecture_hash")
        if isinstance(rows[0].get("artifacts"), dict)
        else None,
        metrics=first_parent,
        graph=first_parent_code,
        input_tokens=0,
        output_tokens=0,
        failure_reason="",
        infrastructure_failure=False,
    )
    provider_attempts_path = run_dir / "controller" / "provider_attempts.jsonl"
    provider_attempts = load_jsonl(provider_attempts_path) if provider_attempts_path.exists() else []
    usage_by_ordinal = {
        int(row["attempt_ordinal"]): row
        for row in provider_attempts
        if isinstance(row.get("attempt_ordinal"), int)
    }
    trace_by_iteration = {int(row["iteration"]): row for row in rows}
    outcomes_path = run_dir / "controller" / "proposal_terminal_outcomes.jsonl"
    outcomes = load_jsonl(outcomes_path) if outcomes_path.exists() else []
    expected_iterations = sorted(
        int(row["iteration"])
        for row in outcomes
        if isinstance(row.get("iteration"), int)
    )
    if not expected_iterations:
        expected_iterations = sorted(trace_by_iteration)

    candidates: list[dict[str, Any]] = []
    for iteration in expected_iterations:
        row = trace_by_iteration.get(iteration)
        if row is None:
            # OpenEvolve's asynchronous tracer can omit a completed event even
            # though the authoritative terminal ledger and checkpoint contain
            # it. Recover that proposal from the exact iteration checkpoint.
            checkpoint = run_dir / "controller" / "checkpoints" / f"checkpoint_{iteration}" / "programs"
            recovered = []
            for path in sorted(checkpoint.glob("*.json")):
                program = load_json(path)
                if program.get("iteration_found") == iteration:
                    recovered.append(program)
            if len(recovered) != 1:
                raise ValueError(
                    f"{run_dir.name}: trace omits iteration {iteration} and "
                    f"checkpoint recovery found {len(recovered)} candidates"
                )
            program = recovered[0]
            artifacts = parse_json_object(program.get("artifacts_json"))
            metrics = program.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
            usage = usage_by_ordinal.get(iteration, {})
            candidates.append(
                candidate_record(
                    iteration=iteration,
                    candidate_id=str(program.get("id") or f"proposal-{iteration}"),
                    parent_id=str(program["parent_id"]) if program.get("parent_id") else None,
                    architecture_hash=str(artifacts.get("candidate_architecture_hash") or "") or None,
                    metrics=metrics,
                    graph=parse_graph(program.get("code")),
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    failure_reason=str(artifacts.get("failure_stage") or ""),
                    infrastructure_failure=bool(artifacts.get("infrastructure_failure")),
                )
            )
            continue

        metrics = row.get("child_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        usage = usage_by_ordinal.get(iteration, {})
        stderr = artifacts.get("stderr")
        failure_reason = str(stderr).splitlines()[0] if isinstance(stderr, str) else ""
        candidates.append(
            candidate_record(
                iteration=iteration,
                candidate_id=str(row.get("child_id") or f"proposal-{iteration}"),
                parent_id=str(row["parent_id"]) if row.get("parent_id") else None,
                architecture_hash=str(artifacts.get("candidate_architecture_hash") or "") or None,
                metrics=metrics,
                graph=parse_graph(row.get("child_code") or row.get("llm_response")),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                failure_reason=failure_reason,
                infrastructure_failure=bool(artifacts.get("infrastructure_failure")),
            )
        )
    return seed, candidates


def complete_run_candidates(run_dir: Path, method: str) -> int:
    try:
        if method == "ar":
            return max(0, len(load_jsonl(run_dir / "controller" / "lineage.jsonl")) - 1)
        result = load_json(run_dir / "controller" / "run_result.json")
        completed = result.get("proposal_opportunities_completed")
        return int(completed) if isinstance(completed, int) and result.get("completed") is True else -1
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return -1


def discover_runs(downloads_root: Path) -> list[Path]:
    grouped: dict[tuple[str, int, str], list[Path]] = {}
    for path in downloads_root.iterdir():
        if not path.is_dir():
            continue
        match = RUN_PATTERN.match(path.name)
        if match is None:
            continue
        key = (match.group("method"), int(match.group("horizon")), f"RD{match.group('condition')}")
        grouped.setdefault(key, []).append(path)
    selected: list[Path] = []
    for key, paths in grouped.items():
        method, horizon, _condition = key
        ranked = sorted(
            paths,
            key=lambda path: (
                complete_run_candidates(path, method),
                int(RUN_PATTERN.match(path.name).group("retry") or 0),
                path.name,
            ),
            reverse=True,
        )
        best = ranked[0]
        if complete_run_candidates(best, method) != horizon:
            raise ValueError(
                f"no complete {method} h{horizon} {_condition} run; best was "
                f"{best.name} with {complete_run_candidates(best, method)} proposals"
            )
        selected.append(best)
    return sorted(
        selected,
        key=lambda path: (
            int(RUN_PATTERN.match(path.name).group("horizon")),
            RUN_PATTERN.match(path.name).group("method"),
            int(RUN_PATTERN.match(path.name).group("condition")),
        ),
    )


def enrich_trajectory(seed: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = seed["parameterCount"] if seed["valid"] else None
    seen = {seed["architectureHash"]} if seed.get("architectureHash") else set()
    cumulative_tokens = 0
    trajectory: list[dict[str, Any]] = [{**seed, "bestValidParameterCount": best, "isNewArchitecture": True, "cumulativeTokens": 0}]
    for candidate in candidates:
        params = candidate["parameterCount"]
        if candidate["valid"] and params is not None:
            best = params if best is None else min(best, params)
        architecture_hash = candidate.get("architectureHash")
        is_new = architecture_hash is not None and architecture_hash not in seen
        if architecture_hash is not None:
            seen.add(architecture_hash)
        cumulative_tokens += candidate["inputTokens"] + candidate["outputTokens"]
        trajectory.append(
            {
                **candidate,
                "bestValidParameterCount": best,
                "isNewArchitecture": is_new,
                "cumulativeTokens": cumulative_tokens,
            }
        )
    return trajectory


def build_run(run_dir: Path) -> dict[str, Any]:
    match = RUN_PATTERN.match(run_dir.name)
    if match is None:
        raise ValueError(run_dir.name)
    method_code = match.group("method")
    horizon = int(match.group("horizon"))
    condition = f"RD{match.group('condition')}"
    config = load_json(run_dir / "controller" / "research_process" / "study_config.json")
    seed, candidates = ar_candidates(run_dir) if method_code == "ar" else oe_candidates(run_dir)
    if len(candidates) != horizon:
        raise ValueError(f"{run_dir.name}: expected {horizon} proposals, found {len(candidates)}")
    trajectory = enrich_trajectory(seed, candidates)
    proposal_trajectory = trajectory[1:]
    valid = [row for row in proposal_trajectory if row["valid"]]
    unique = [row for row in proposal_trajectory if row["isNewArchitecture"]]
    improvements = sum(
        1
        for prior, current in zip(trajectory, trajectory[1:])
        if current["bestValidParameterCount"] is not None
        and prior["bestValidParameterCount"] is not None
        and current["bestValidParameterCount"] < prior["bestValidParameterCount"]
    )
    final_best = trajectory[-1]["bestValidParameterCount"]
    return {
        "id": run_dir.name,
        "method": "AutoResearch" if method_code == "ar" else "OpenEvolve",
        "methodCode": method_code,
        "horizon": horizon,
        "condition": condition,
        "conditionLabel": CONDITION_LABELS[condition],
        "memoryPolicy": config["condition"]["memory_policy"],
        "deliberationPolicy": config["condition"]["deliberation_policy"],
        "proposalCount": len(candidates),
        "validCount": len(valid),
        "uniqueArchitectureCount": len(unique),
        "improvementCount": improvements,
        "finalBestParameterCount": final_best,
        "finalBestPublicAccuracy": next(
            (
                row["publicAccuracy"]
                for row in reversed(trajectory)
                if row["valid"] and row["parameterCount"] == final_best
            ),
            None,
        ),
        "trajectory": trajectory,
    }


def markdown_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def format_params(value: Any) -> str:
    return "—" if value is None else f"{int(value):,}"


def build_report(payload: dict[str, Any]) -> str:
    runs = payload["runs"]
    proposal_total = sum(run["proposalCount"] for run in runs)
    lines = [
        "# Architecture Search Trajectories: 8-, 12-, and 20-Iteration Sweep",
        "",
        "This file contains the complete proposal-level results for the 24-run process-intervention sweep: two research frameworks × four memory/deliberation conditions × three search horizons.",
        "",
        "## What was measured",
        "",
        f"- **Runs:** {len(runs)}",
        f"- **Generated proposals:** {proposal_total} (seed architectures are shown separately at iteration 0 and are not counted as proposals)",
        "- **Primary trajectory:** the smallest parameter count among candidates that executed successfully, passed the transformer-validity check, and remained eligible as a parent after each proposal",
        "- **RD0:** sequential memory + neutral review",
        "- **RD1:** sequential memory + assumption challenge",
        "- **RD2:** portfolio memory + neutral review",
        "- **RD3:** portfolio memory + assumption challenge",
        "",
        "> Important interpretation: validity here is structural/execution validity, not task competence. Public accuracy is reported independently. A tiny zero-accuracy architecture can therefore become the parameter-count incumbent; this is evidence of objective collapse, not a successful AdderBoard model.",
        "",
        "> Data provenance: OpenEvolve h8/RD0 iteration 6 is present in the authoritative terminal-outcome ledger and checkpoint, but its asynchronous `evolution_trace.jsonl` event was omitted. This report reconstructs that proposal from `checkpoint_6`; no metrics or architecture fields were imputed.",
        "",
        "## Run summary",
        "",
        "| Horizon | Framework | Condition | Valid | Unique | Improvements | Final best params | Accuracy of final incumbent | Run ID |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        accuracy = run["finalBestPublicAccuracy"]
        lines.append(
            "| {horizon} | {method} | {condition} | {valid}/{total} | {unique} | {improvements} | {params} | {accuracy} | `{run_id}` |".format(
                horizon=run["horizon"],
                method=run["method"],
                condition=run["condition"],
                valid=run["validCount"],
                total=run["proposalCount"],
                unique=run["uniqueArchitectureCount"],
                improvements=run["improvementCount"],
                params=format_params(run["finalBestParameterCount"]),
                accuracy="—" if accuracy is None else f"{accuracy:.4f}",
                run_id=run["id"],
            )
        )
    lines.extend(
        [
            "",
            "## Complete trajectories",
            "",
            "Each table includes the shared seed at iteration 0 followed by every generated proposal. `Best valid` is a running minimum and is carried forward when a proposal is invalid.",
            "",
        ]
    )
    for run in runs:
        lines.extend(
            [
                f"### {run['method']} · h{run['horizon']} · {run['condition']}",
                "",
                f"- Run: `{run['id']}`",
                f"- Condition: {run['conditionLabel']}",
                "",
                "| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |",
                "|---:|---|---:|---:|:---:|:---:|---:|---|---|---|",
            ]
        )
        for row in run["trajectory"]:
            candidate = row["architectureHash"] or row["candidateId"]
            accuracy = row["publicAccuracy"]
            lines.append(
                "| {iteration} | `{candidate}` | {params} | {best} | {valid} | {unique} | {accuracy} | {name} | {mechanism} | {failure} |".format(
                    iteration=row["iteration"],
                    candidate=markdown_escape(candidate)[:12],
                    params=format_params(row["parameterCount"]),
                    best=format_params(row["bestValidParameterCount"]),
                    valid="yes" if row["valid"] else "no",
                    unique="yes" if row["isNewArchitecture"] else "no",
                    accuracy="—" if accuracy is None else f"{accuracy:.4f}",
                    name=markdown_escape(row["architectureName"]),
                    mechanism=markdown_escape(row["mechanismHypothesis"]),
                    failure=markdown_escape(row["failureReason"]),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Reading the dashboard",
            "",
            "The dashboard plots iteration on the x-axis and the running best structurally valid parameter count on the y-axis. Toggle horizons, frameworks, and intervention conditions to compare search dynamics rather than only terminal winners. The proposal table exposes the underlying architectures and failure/duplication events.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_payload(payload: dict[str, Any], *, require_complete: bool) -> None:
    runs = payload["runs"]
    expected = 24 if require_complete else len(runs)
    if len(runs) != expected:
        raise ValueError(f"expected {expected} selected runs, found {len(runs)}")
    identities = {(run["methodCode"], run["horizon"], run["condition"]) for run in runs}
    if len(identities) != len(runs):
        raise ValueError("duplicate framework/horizon/condition cells")
    if require_complete:
        expected_identities = {
            (method, horizon, condition)
            for method in ("ar", "oe")
            for horizon in (8, 12, 20)
            for condition in CONDITION_LABELS
        }
        if identities != expected_identities:
            raise ValueError("the 24-cell experiment matrix is incomplete")
        if sum(run["proposalCount"] for run in runs) != 320:
            raise ValueError("the complete sweep must contain exactly 320 proposals")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    selected = discover_runs(args.downloads_root)
    runs = [build_run(path) for path in selected]
    payload = {
        "schemaVersion": "1.0",
        "runCount": len(runs),
        "proposalCount": sum(run["proposalCount"] for run in runs),
        "runs": runs,
    }
    validate_payload(payload, require_complete=not args.allow_partial)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(build_report(payload), encoding="utf-8")
    print(
        f"wrote {len(runs)} runs and {payload['proposalCount']} proposals to "
        f"{args.output_json} and {args.report}"
    )


if __name__ == "__main__":
    main()
