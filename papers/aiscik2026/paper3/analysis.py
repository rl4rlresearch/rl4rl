#!/usr/bin/env python3
# ruff: noqa: E501
"""Reproduce Paper 3's population-memory and path-dependence analysis.

The primary corpus is 64 unified-v3 Tiny AdderBoard trajectories.  Every run
has a complete 70-opportunity prefix.  The greedy engine exposes either a
single incumbent (C0/C1) or a K=4 controller portfolio (C2/C3).  The native
engine delegates selection, retention, MAP-Elites archiving, and five-island
population management to the vendored OpenEvolve ProgramDatabase; this
population layer supersedes the nominal controller-memory label, so all native
conditions are analyzed as one memory architecture and prompt policy is kept
as a stratification variable.

The completed 20-run, 200-opportunity Fashion-MNIST greedy campaign is a
cross-task descriptive replication of the K=1 versus K=4 controller contrast.
No analysis uses events after the common Tiny horizon of 70.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import itertools
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_OUTPUT = HERE / "derived"
GREEDY = REPO / "data/c0c3/unified-v3-tiny-adderboard-greedy-campaign"
NATIVE = REPO / "data/c0c3/unified-v3-tiny-adderboard-native-campaign"
FASHION = REPO / "data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign"
TINY_HORIZON = 70
FASHION_HORIZON = 200
SEED = 20260901
REFERENCE_ATTRIBUTION_RE = re.compile(
    r"\breference design(?:s)?\b|\balternative design(?:s)?\b|\bavailable design(?:s)?\b",
    re.I,
)
BROAD_REFERENCE_ATTRIBUTION_RE = re.compile(
    r"reference design|reference|alternative design|available design", re.I
)
COMPARATIVE_EVIDENCE_RE = re.compile(
    r"while|whereas|compared|best|outperform|failed|earlier|prior|current", re.I
)
NUMERIC_EVIDENCE_RE = re.compile(r"\d")


def source_delta_signature(run_dir: Path, baseline_id: str, candidate_id: str) -> set[str]:
    baseline_path = run_dir / "candidates" / baseline_id / "train.py"
    candidate_path = run_dir / "candidates" / candidate_id / "train.py"
    if not baseline_path.is_file() or not candidate_path.is_file():
        return set()
    baseline = baseline_path.read_text(encoding="utf-8", errors="replace").splitlines()
    candidate = candidate_path.read_text(encoding="utf-8", errors="replace").splitlines()
    signature: set[str] = set()
    for line in difflib.unified_diff(baseline, candidate, n=0):
        if not (line.startswith("+") or line.startswith("-")):
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "#", line[1:].strip())
        if normalized:
            signature.add(line[0] + normalized)
    return signature


def mean_pairwise_jaccard_distance(signatures: list[set[str]]) -> float:
    usable = [value for value in signatures if value]
    if len(usable) < 2:
        return math.nan
    distances: list[float] = []
    for left, right in itertools.combinations(usable, 2):
        union = left | right
        distances.append(1.0 - len(left & right) / len(union) if union else 0.0)
    return statistics.mean(distances)


@dataclass(frozen=True)
class Run:
    task: str
    architecture: str
    campaign: Path
    run_dir: Path
    run_id: str
    block: int
    condition: str
    prompt_policy: str
    nominal_memory: str
    memory_system: str
    horizon: int
    baseline: dict[str, Any]
    state: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    starts: tuple[dict[str, Any], ...]
    native_events: tuple[dict[str, Any], ...]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL: {path}:{number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected object: {path}:{number}")
        output.append(value)
    return output


def load_runs(
    campaign: Path,
    *,
    task: str,
    architecture: str,
    horizon: int,
) -> list[Run]:
    runs: list[Run] = []
    for run_dir in sorted((campaign / "runs").iterdir()):
        if not run_dir.is_dir() or not (run_dir / "manifest.json").is_file():
            continue
        manifest = load_json(run_dir / "manifest.json")
        assignment = manifest["assignment"]
        condition = str(assignment["condition"])
        rows = load_jsonl(run_dir / "events.jsonl")
        events = tuple(
            sorted(
                (
                    row
                    for row in rows
                    if row.get("event") == "proposal_completed"
                    and int(row.get("opportunity", 0)) <= horizon
                ),
                key=lambda row: int(row["opportunity"]),
            )
        )
        starts = tuple(
            sorted(
                (
                    row
                    for row in rows
                    if row.get("event") == "proposal_started"
                    and int(row.get("opportunity", 0)) <= horizon
                ),
                key=lambda row: int(row["opportunity"]),
            )
        )
        native_path = run_dir / "native-openevolve/events.jsonl"
        native_events = tuple(load_jsonl(native_path)) if native_path.is_file() else ()
        nominal = "portfolio" if condition in {"C2", "C3"} else "single"
        memory_system = (
            "native_population"
            if architecture == "native"
            else ("greedy_portfolio" if nominal == "portfolio" else "greedy_single")
        )
        runs.append(
            Run(
                task=task,
                architecture=architecture,
                campaign=campaign,
                run_dir=run_dir,
                run_id=str(assignment["run_id"]),
                block=int(assignment["block"]),
                condition=condition,
                prompt_policy=("challenge" if condition in {"C1", "C3"} else "ordinary"),
                nominal_memory=nominal,
                memory_system=memory_system,
                horizon=horizon,
                baseline=manifest["baseline"],
                state=load_json(run_dir / "state.json"),
                events=events,
                starts=starts,
                native_events=native_events,
            )
        )
    return runs


def usage_total(value: dict[str, Any] | None) -> int:
    value = value or {}
    if value.get("total_tokens") is not None:
        return int(value.get("total_tokens") or 0)
    return int(value.get("input_tokens") or 0) + int(value.get("output_tokens") or 0)


def event_map(run: Run) -> dict[int, dict[str, Any]]:
    return {int(row["opportunity"]): row for row in run.events}


def candidate_table(run: Run) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    root_id = str(run.baseline["candidate_id"])
    table[root_id] = {
        "candidate_id": root_id,
        "parent_ids": [],
        "created_opportunity": 0,
        "metrics": run.baseline.get("metrics") or {},
        "fitness": run.baseline.get("fitness"),
    }
    for candidate_id, value in (run.state.get("candidates") or {}).items():
        table[str(candidate_id)] = dict(value)
    for event in run.events:
        candidate_id = event.get("candidate_id")
        if not candidate_id:
            continue
        evaluation = event.get("evaluation") or {}
        table.setdefault(
            str(candidate_id),
            {
                "candidate_id": str(candidate_id),
                "parent_ids": list(event.get("parent_ids") or event.get("selected_parent_ids") or []),
                "created_opportunity": int(event["opportunity"]),
                "metrics": (evaluation.get("metrics") or {}),
                "fitness": evaluation.get("fitness"),
            },
        )
    return table


def parent_id(event: dict[str, Any]) -> str:
    values = event.get("selected_parent_ids") or event.get("parent_ids") or []
    return str(values[0]) if values else ""


def top_branch(
    candidate_id: str,
    *,
    root_id: str,
    candidates: dict[str, dict[str, Any]],
    cache: dict[str, str],
) -> str:
    if not candidate_id or candidate_id == root_id:
        return root_id
    if candidate_id in cache:
        return cache[candidate_id]
    visited: set[str] = set()
    current = candidate_id
    while current and current != root_id:
        if current in visited:
            cache[candidate_id] = current
            return current
        visited.add(current)
        value = candidates.get(current) or {}
        parents = value.get("parent_ids") or []
        if not parents:
            cache[candidate_id] = current
            return current
        parent = str(parents[0])
        if parent == root_id:
            cache[candidate_id] = current
            return current
        current = parent
    cache[candidate_id] = root_id
    return root_id


def shannon_effective(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return math.exp(entropy)


def normalized_entropy(values: list[str]) -> float:
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    total = len(values)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return entropy / math.log(len(counts))


def candidate_objective(run: Run, candidate_id: str, table: dict[str, dict[str, Any]]) -> float:
    value = table.get(candidate_id) or {}
    metrics = value.get("metrics") or {}
    if run.task == "tiny_adderboard":
        parameters = metrics.get("parameters")
        return -float(parameters) if parameters is not None else math.nan
    fitness = value.get("fitness")
    if fitness is None:
        fitness = metrics.get("validation_score")
    return float(fitness) if fitness is not None else math.nan


def incumbent_objective_series(run: Run, table: dict[str, dict[str, Any]]) -> list[float]:
    output: list[float] = []
    for event in run.events:
        output.append(candidate_objective(run, str(event.get("incumbent_after") or ""), table))
    return output


def trajectory_row(run: Run) -> dict[str, Any]:
    if len(run.events) != run.horizon:
        raise ValueError(f"{run.run_id}: expected {run.horizon} completions, found {len(run.events)}")
    opportunities = [int(row["opportunity"]) for row in run.events]
    if opportunities != list(range(1, run.horizon + 1)):
        raise ValueError(f"{run.run_id}: non-contiguous opportunities")
    table = candidate_table(run)
    root_id = str(run.baseline["candidate_id"])
    cache: dict[str, str] = {}
    objectives = incumbent_objective_series(run, table)
    baseline_objective = candidate_objective(run, root_id, table)
    final_objective = objectives[-1]
    if not (math.isfinite(baseline_objective) and math.isfinite(final_objective)):
        raise ValueError(f"{run.run_id}: objective unavailable")
    if run.task == "tiny_adderboard":
        baseline_parameters = -baseline_objective
        final_parameters = -final_objective
        normalized_gain = (baseline_parameters - final_parameters) / baseline_parameters
        auc_gain = statistics.mean((baseline_parameters + value) / baseline_parameters for value in objectives)
    else:
        baseline_parameters = math.nan
        final_parameters = math.nan
        normalized_gain = (final_objective - baseline_objective) / abs(baseline_objective)
        auc_gain = statistics.mean((value - baseline_objective) / abs(baseline_objective) for value in objectives)

    selected_parents: list[str] = []
    selected_branches: list[str] = []
    postwarm_branches: list[str] = []
    postwarm_parents: list[str] = []
    parent_ages: list[int] = []
    exact_reactivations = 0
    branch_reactivations = 0
    productive_branch_reactivations = 0
    alternative_parent_selections = 0
    alternative_parent_improvements = 0
    global_improvements = 0
    reference_attributions = 0
    broad_reference_attributions = 0
    comparative_evidence = 0
    numeric_evidence = 0
    evidence_characters = 0
    last_parent_seen: dict[str, int] = {}
    last_branch_seen: dict[str, int] = {}
    branch_first: dict[str, int] = {}
    branch_last: dict[str, int] = {}
    branch_selection_counts: Counter[str] = Counter()
    qualified = 0
    retained = 0
    last_improvement = 0
    incumbent_before = root_id
    event_details: list[dict[str, Any]] = []

    for event in run.events:
        opportunity = int(event["opportunity"])
        selected = parent_id(event)
        branch = top_branch(selected, root_id=root_id, candidates=table, cache=cache)
        selected_parents.append(selected)
        selected_branches.append(branch)
        if opportunity >= 10:
            postwarm_parents.append(selected)
            if branch != root_id:
                postwarm_branches.append(branch)
        created = int((table.get(selected) or {}).get("created_opportunity") or 0)
        parent_age = max(0, opportunity - created)
        parent_ages.append(parent_age)
        exact_gap = opportunity - last_parent_seen[selected] if selected in last_parent_seen else 0
        branch_gap = opportunity - last_branch_seen[branch] if branch in last_branch_seen else 0
        exact_reactivation = bool(exact_gap >= 10)
        branch_reactivation = bool(branch_gap >= 10)
        exact_reactivations += int(exact_reactivation)
        branch_reactivations += int(branch_reactivation)
        alternative = bool(selected and selected != incumbent_before)
        alternative_parent_selections += int(alternative)
        evaluation = event.get("evaluation") or {}
        evidence_text = str(event.get("evidence") or "")
        reference_attributions += int(bool(REFERENCE_ATTRIBUTION_RE.search(evidence_text)))
        broad_reference_attributions += int(
            bool(BROAD_REFERENCE_ATTRIBUTION_RE.search(evidence_text))
        )
        comparative_evidence += int(bool(COMPARATIVE_EVIDENCE_RE.search(evidence_text)))
        numeric_evidence += int(bool(NUMERIC_EVIDENCE_RE.search(evidence_text)))
        evidence_characters += len(evidence_text)
        qualified += int(bool(evaluation.get("valid")))
        retained += int(bool(event.get("retained")))
        incumbent_after = str(event.get("incumbent_after") or incumbent_before)
        before_objective = candidate_objective(run, incumbent_before, table)
        after_objective = candidate_objective(run, incumbent_after, table)
        improved = bool(
            math.isfinite(before_objective)
            and math.isfinite(after_objective)
            and after_objective > before_objective + 1e-12
        )
        if improved:
            global_improvements += 1
            last_improvement = opportunity
        alternative_parent_improvements += int(alternative and improved)
        productive_branch_reactivations += int(branch_reactivation and improved)
        last_parent_seen[selected] = opportunity
        last_branch_seen[branch] = opportunity
        branch_first.setdefault(branch, opportunity)
        branch_last[branch] = opportunity
        branch_selection_counts[branch] += 1
        event_details.append(
            {
                "opportunity": opportunity,
                "candidate_id": str(event.get("candidate_id") or ""),
                "parent_id": selected,
                "branch_id": branch,
                "parent_age": parent_age,
                "exact_parent_gap": exact_gap,
                "branch_gap": branch_gap,
                "exact_reactivation": int(exact_reactivation),
                "branch_reactivation": int(branch_reactivation),
                "alternative_parent": int(alternative),
                "global_improvement": int(improved),
                "incumbent_before": incumbent_before,
                "incumbent_after": incumbent_after,
                "mechanism": str(event.get("mechanism") or ""),
                "hypothesis": str(event.get("hypothesis") or ""),
                "evidence": evidence_text,
                "retained": int(bool(event.get("retained"))),
            }
        )
        incumbent_before = incumbent_after

    nonroot_branches = [value for value in postwarm_branches if value != root_id]
    branch_delta_diversity = mean_pairwise_jaccard_distance(
        [
            source_delta_signature(run.run_dir, root_id, branch_id)
            for branch_id in sorted(set(nonroot_branches))
        ]
    )
    branch_spans = [
        branch_last[value] - branch_first[value]
        for value in branch_first
        if value != root_id
    ]
    visible_counts = [len(row.get("visible_candidate_ids") or []) for row in run.starts]
    portfolio_occupancy = [len(row.get("portfolio_after") or []) for row in run.events]
    token_total = sum(usage_total(row.get("usage_increment")) for row in run.events)
    output_tokens = sum(int((row.get("usage_increment") or {}).get("output_tokens") or 0) for row in run.events)
    evaluator_seconds = sum(float(row.get("evaluator_seconds_increment") or 0.0) for row in run.events)

    native_outcomes = [
        row
        for row in run.native_events
        if row.get("event") in {"native_outcome_committed", "native_outcome_reconciled_without_addition"}
        and int(row.get("opportunity", 0)) <= run.horizon
    ]
    native_samples = [
        row
        for row in run.native_events
        if row.get("event") == "native_parent_sampled"
        and int(row.get("opportunity", 0)) <= run.horizon
    ]
    native_population_final = (
        float(next((row.get("population_size") for row in reversed(native_outcomes) if row.get("population_size") is not None), math.nan))
        if native_outcomes
        else math.nan
    )
    native_archive_final = (
        float(next((row.get("archive_size") for row in reversed(native_outcomes) if row.get("archive_size") is not None), math.nan))
        if native_outcomes
        else math.nan
    )
    native_admission_rate = (
        statistics.mean(bool(row.get("candidate_in_population")) for row in native_outcomes)
        if native_outcomes
        else math.nan
    )
    native_inspiration_mean = (
        statistics.mean(len(row.get("inspiration_ids") or []) for row in native_samples)
        if native_samples
        else math.nan
    )
    native_islands_occupied = (
        statistics.mean(sum(int(value > 0) for value in (row.get("island_sizes") or [])) for row in native_outcomes if row.get("island_sizes"))
        if any(row.get("island_sizes") for row in native_outcomes)
        else math.nan
    )

    row = {
        "task": run.task,
        "architecture": run.architecture,
        "run_id": run.run_id,
        "block": run.block,
        "condition": run.condition,
        "prompt_policy": run.prompt_policy,
        "nominal_memory": run.nominal_memory,
        "memory_system": run.memory_system,
        "horizon": run.horizon,
        "baseline_objective": baseline_objective,
        "final_objective": final_objective,
        "normalized_gain": normalized_gain,
        "auc_normalized_gain": auc_gain,
        "baseline_parameters": baseline_parameters,
        "final_parameters": final_parameters,
        "qualified_rate": qualified / run.horizon,
        "retained_rate": retained / run.horizon,
        "global_improvements": global_improvements,
        "tail_stagnation": run.horizon - last_improvement,
        "reference_attribution_rate": reference_attributions / run.horizon,
        "broad_reference_attribution_rate": (
            broad_reference_attributions / run.horizon
        ),
        "comparative_evidence_rate": comparative_evidence / run.horizon,
        "numeric_evidence_rate": numeric_evidence / run.horizon,
        "mean_evidence_characters": evidence_characters / run.horizon,
        "unique_selected_parents": len(set(postwarm_parents)),
        "effective_selected_parents": shannon_effective(postwarm_parents),
        "parent_selection_entropy": normalized_entropy(postwarm_parents),
        "unique_top_lineages": len(set(nonroot_branches)),
        "effective_top_lineages": shannon_effective(nonroot_branches),
        "branch_delta_diversity": branch_delta_diversity,
        "dominant_lineage_share": (max(Counter(nonroot_branches).values()) / len(nonroot_branches) if nonroot_branches else 1.0),
        "long_lived_lineages": sum(span >= 10 for span in branch_spans),
        "max_lineage_span": max(branch_spans, default=0),
        "mean_parent_age": statistics.mean(parent_ages),
        "old_parent_rate": statistics.mean(age >= 10 for age in parent_ages),
        "exact_parent_reactivations": exact_reactivations,
        "branch_reactivations": branch_reactivations,
        "productive_branch_reactivations": productive_branch_reactivations,
        "alternative_parent_rate": alternative_parent_selections / run.horizon,
        "alternative_parent_improvements": alternative_parent_improvements,
        "improvement_from_alternative_share": (alternative_parent_improvements / global_improvements if global_improvements else 0.0),
        "mean_visible_designs": statistics.mean(visible_counts) if visible_counts else math.nan,
        "max_visible_designs": max(visible_counts, default=0),
        "mean_portfolio_occupancy": statistics.mean(portfolio_occupancy) if portfolio_occupancy else math.nan,
        "tokens": token_total,
        "tokens_per_proposal": token_total / run.horizon,
        "tokens_per_global_improvement": token_total / global_improvements if global_improvements else math.nan,
        "output_tokens": output_tokens,
        "evaluator_seconds": evaluator_seconds,
        "native_population_final": native_population_final,
        "native_archive_final": native_archive_final,
        "native_admission_rate": native_admission_rate,
        "native_inspiration_mean": native_inspiration_mean,
        "native_islands_occupied_mean": native_islands_occupied,
    }
    row["event_details"] = event_details
    row["selected_branches"] = selected_branches
    return row


def mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.mean(finite) if finite else math.nan


def percentile_interval(values: list[float], low: float = 2.5, high: float = 97.5) -> tuple[float, float]:
    return float(np.percentile(values, low)), float(np.percentile(values, high))


def bootstrap_mean_ci(rows: list[dict[str, Any]], metric: str, *, seed: int, samples: int = 20000) -> tuple[float, float]:
    blocks = sorted({int(row["block"]) for row in rows})
    by_block = {block: [row for row in rows if int(row["block"]) == block] for block in blocks}
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        chosen = [rng.choice(blocks) for _ in blocks]
        sample = [row for block in chosen for row in by_block[block]]
        values.append(mean([float(row[metric]) for row in sample]))
    return percentile_interval(values)


def paired_contrast(
    rows: list[dict[str, Any]],
    *,
    left_condition: str,
    right_condition: str,
    metric: str,
    seed: int,
    samples: int = 20000,
) -> dict[str, Any]:
    lookup = {(int(row["block"]), str(row["condition"])): row for row in rows}
    blocks = sorted({int(row["block"]) for row in rows})
    differences = [
        float(lookup[(block, right_condition)][metric]) - float(lookup[(block, left_condition)][metric])
        for block in blocks
    ]
    rng = random.Random(seed)
    boot = [statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(samples)]
    low, high = percentile_interval(boot)
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    ties = len(differences) - positive - negative
    return {
        "metric": metric,
        "left_condition": left_condition,
        "right_condition": right_condition,
        "n_pairs": len(differences),
        "left_mean": mean([float(lookup[(block, left_condition)][metric]) for block in blocks]),
        "right_mean": mean([float(lookup[(block, right_condition)][metric]) for block in blocks]),
        "paired_difference": statistics.mean(differences),
        "bootstrap_low": low,
        "bootstrap_high": high,
        "right_higher": positive,
        "left_higher": negative,
        "ties": ties,
        "differences": differences,
    }


def group_summaries(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["memory_system"]), str(row["prompt_policy"]))].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        record: dict[str, Any] = {
            "task": key[0],
            "memory_system": key[1],
            "prompt_policy": key[2],
            "n": len(values),
        }
        for index, metric in enumerate(metrics):
            metric_values = [float(row[metric]) for row in values]
            finite_values = [value for value in metric_values if math.isfinite(value)]
            record[f"{metric}_mean"] = mean(metric_values)
            record[f"{metric}_median"] = (
                float(np.median(finite_values)) if finite_values else math.nan
            )
            low, high = bootstrap_mean_ci(values, metric, seed=SEED + index)
            record[f"{metric}_low"] = low
            record[f"{metric}_high"] = high
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields and key not in {"event_details", "selected_branches", "differences"}:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fields} for row in rows)


def source_audit(run: Run, parent: str, candidate: str) -> dict[str, Any]:
    parent_path = run.run_dir / "candidates" / parent / "train.py"
    candidate_path = run.run_dir / "candidates" / candidate / "train.py"
    if not parent_path.is_file() or not candidate_path.is_file():
        return {
            "source_available": 0,
            "changed_lines": math.nan,
            "parent_source_sha256": "",
            "candidate_source_sha256": "",
        }
    parent_bytes = parent_path.read_bytes()
    candidate_bytes = candidate_path.read_bytes()
    changes = [
        line
        for line in difflib.unified_diff(
            parent_bytes.decode("utf-8", errors="replace").splitlines(),
            candidate_bytes.decode("utf-8", errors="replace").splitlines(),
            n=0,
        )
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    ]
    return {
        "source_available": 1,
        "changed_lines": len(changes),
        "parent_source_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
    }


def label_system(value: str) -> str:
    return {
        "greedy_single": "Greedy K=1",
        "greedy_portfolio": "Greedy K=4",
        "native_population": "Native population",
    }[value]


COLORS = {
    "greedy_single": "#486A8C",
    "greedy_portfolio": "#C87941",
    "native_population": "#477A5B",
}


def plot_system_summary(rows: list[dict[str, Any]], path: Path) -> None:
    tiny = [row for row in rows if row["task"] == "tiny_adderboard"]
    systems = ["greedy_single", "greedy_portfolio", "native_population"]
    prompts = ["ordinary", "challenge"]
    metrics = [
        ("normalized_gain", "Endpoint parameter reduction", "fraction of baseline"),
        (
            "effective_top_lineages",
            "Effective selected branches (manipulation check)",
            "exp(Shannon entropy)",
        ),
        ("alternative_parent_rate", "Non-incumbent parent use", "fraction of proposals"),
        ("tokens_per_proposal", "Subject-agent token cost", "tokens / proposal"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.0))
    rng = np.random.default_rng(SEED)
    for ax, (metric, title, ylabel) in zip(axes.flat, metrics, strict=True):
        positions: list[float] = []
        labels: list[str] = []
        for index, system in enumerate(systems):
            for offset, prompt in enumerate(prompts):
                x = index * 2.5 + offset * 0.75
                values = [float(row[metric]) for row in tiny if row["memory_system"] == system and row["prompt_policy"] == prompt]
                positions.append(x)
                labels.append("Ord." if prompt == "ordinary" else "Chal.")
                jitter = rng.normal(0, 0.055, len(values))
                ax.scatter(np.full(len(values), x) + jitter, values, s=17, alpha=0.48, color=COLORS[system], edgecolors="none")
                summary = mean(values)
                low, high = bootstrap_mean_ci(
                    [row for row in tiny if row["memory_system"] == system and row["prompt_policy"] == prompt],
                    metric,
                    seed=SEED + index + offset,
                    samples=5000,
                )
                ax.errorbar(x, summary, yerr=[[summary - low], [high - summary]], fmt="o", ms=6, color="#18212B", capsize=3, lw=1.3)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(positions, labels, fontsize=7)
        for index, system in enumerate(systems):
            ax.text(index * 2.5 + 0.375, -0.25, label_system(system), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color=COLORS[system], fontweight="bold")
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(hspace=0.52, wspace=0.35, bottom=0.17)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_lineage_raster(runs: list[Run], rows: list[dict[str, Any]], path: Path) -> None:
    wanted = [
        ("greedy", 1, "C0", "Greedy K=1"),
        ("greedy", 1, "C2", "Greedy K=4"),
        ("native", 1, "C2", "Native population"),
    ]
    lookup_run = {(run.architecture, run.block, run.condition): run for run in runs}
    lookup_row = {
        (row["task"], row["architecture"], int(row["block"]), row["condition"]): row
        for row in rows
    }
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 4.8), sharex=True)
    for ax, (architecture, block, condition, title) in zip(axes, wanted, strict=True):
        run = lookup_run[(architecture, block, condition)]
        row = lookup_row[(run.task, architecture, block, condition)]
        branches = list(row["selected_branches"])
        root = str(run.baseline["candidate_id"])
        labels: dict[str, int] = {root: 0}
        for branch in branches:
            if branch not in labels:
                labels[branch] = len(labels)
        y = [labels[value] for value in branches]
        colors = ["#9AA4AF" if value == root else COLORS[row["memory_system"]] for value in branches]
        ax.scatter(range(1, len(y) + 1), y, c=colors, s=18, marker="s", edgecolors="none")
        for opportunity, value in enumerate(row["event_details"], 1):
            if value["global_improvement"]:
                ax.scatter(opportunity, y[opportunity - 1], s=55, facecolors="none", edgecolors="#111827", linewidths=1.1)
        ax.set_title(title + f" (block {block}, {condition})", loc="left", fontsize=9, fontweight="bold")
        ax.set_ylabel("root lineage", fontsize=8)
        ax.set_yticks(sorted(set(y)))
        ax.grid(axis="x", alpha=0.10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("proposal opportunity (open circle = global incumbent improvement)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_diversity_yield(rows: list[dict[str, Any]], path: Path) -> None:
    tiny = [row for row in rows if row["task"] == "tiny_adderboard"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for system in ["greedy_single", "greedy_portfolio", "native_population"]:
        for prompt, marker in [("ordinary", "o"), ("challenge", "^")]:
            selected = [row for row in tiny if row["memory_system"] == system and row["prompt_policy"] == prompt]
            ax.scatter(
                [row["effective_top_lineages"] for row in selected],
                [row["normalized_gain"] for row in selected],
                s=36,
                marker=marker,
                color=COLORS[system],
                alpha=0.72,
                label=f"{label_system(system)}, {prompt}",
            )
    ax.set_xlabel("effective top-level lineages selected after warm-up", fontsize=9)
    ax.set_ylabel("endpoint parameter reduction (fraction of baseline)", fontsize=9)
    ax.set_title("Balanced branch allocation is not sufficient for task yield", loc="left", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cross_task(contrasts: list[dict[str, Any]], path: Path) -> None:
    chosen = [row for row in contrasts if row["metric"] in {"normalized_gain", "auc_normalized_gain"}]
    fig, ax = plt.subplots(figsize=(8.7, 4.3))
    groups = []
    for task in ["tiny_adderboard", "fashion_mnist"]:
        for prompt in ["ordinary", "challenge"]:
            for metric in ["normalized_gain", "auc_normalized_gain"]:
                groups.append(next(row for row in chosen if row["task"] == task and row["prompt_policy"] == prompt and row["metric"] == metric))
    x = np.arange(len(groups))
    values = [float(row["paired_difference"]) for row in groups]
    lows = [float(row["bootstrap_low"]) for row in groups]
    highs = [float(row["bootstrap_high"]) for row in groups]
    colors = ["#486A8C" if row["task"] == "tiny_adderboard" else "#8A5B9A" for row in groups]
    ax.bar(x, values, color=colors, alpha=0.78, width=0.7)
    ax.errorbar(x, values, yerr=[[value - low for value, low in zip(values, lows, strict=True)], [high - value for value, high in zip(values, highs, strict=True)]], fmt="none", ecolor="#111827", capsize=3, lw=1.1)
    ax.axhline(0, color="#111827", lw=0.9)
    labels = [f"{('Tiny' if row['task']=='tiny_adderboard' else 'Fashion')}\n{row['prompt_policy']}\n{('endpoint' if row['metric']=='normalized_gain' else 'AUC')}" for row in groups]
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("paired K=4 minus K=1 normalized gain", fontsize=9)
    ax.set_title(
        "The controller-portfolio contrast does not transport uniformly",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def trace_exemplars(rows: list[dict[str, Any]], runs: list[Run]) -> list[dict[str, Any]]:
    run_lookup = {run.run_id: run for run in runs}
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["task"] != "tiny_adderboard":
            continue
        run = run_lookup[str(row["run_id"])]
        table = candidate_table(run)
        for detail in row["event_details"]:
            if (
                int(detail["opportunity"]) < 10
                or not detail["alternative_parent"]
                or not detail["global_improvement"]
            ):
                continue
            before = candidate_objective(run, detail["incumbent_before"], table)
            after = candidate_objective(run, detail["incumbent_after"], table)
            reduction = after - before
            audit = source_audit(run, detail["parent_id"], detail["candidate_id"])
            output.append(
                {
                    "architecture": row["architecture"],
                    "memory_system": row["memory_system"],
                    "prompt_policy": row["prompt_policy"],
                    "run_id": row["run_id"],
                    "block": row["block"],
                    "condition": row["condition"],
                    "opportunity": detail["opportunity"],
                    "parent_age": detail["parent_age"],
                    "branch_gap": detail["branch_gap"],
                    "parameter_reduction": reduction,
                    "mechanism": detail["mechanism"],
                    "hypothesis": detail["hypothesis"],
                    **audit,
                }
            )
    output.sort(key=lambda value: (-float(value["parameter_reduction"]), -int(value["parent_age"])))
    return output


def validate_native_semantics(runs: list[Run]) -> dict[str, Any]:
    native = [run for run in runs if run.architecture == "native"]
    if len(native) != 32:
        raise ValueError(f"Expected 32 native runs, found {len(native)}")
    visible_by_condition: dict[str, list[int]] = defaultdict(list)
    prompt_hashes: dict[str, set[str]] = defaultdict(set)
    for run in native:
        if len(run.native_events) == 0:
            raise ValueError(f"{run.run_id}: native events absent")
        samples = [row for row in run.native_events if row.get("event") == "native_parent_sampled" and int(row.get("opportunity", 0)) <= TINY_HORIZON]
        # Fork shadows inherit the leader's immutable opportunities 1--9, but
        # their native event ledger begins at the literal fork (opportunity 10).
        if len(samples) not in {TINY_HORIZON, TINY_HORIZON - 9}:
            raise ValueError(f"{run.run_id}: native sample count {len(samples)}")
        visible_by_condition[run.condition].extend(len(row.get("visible_candidate_ids") or []) for row in run.starts)
        prompt_hashes[run.condition].update(str(row.get("prompt_hashes", {}).get("user", "")) for row in run.events)
    return {
        "native_runs": len(native),
        "mean_visible_by_condition": {key: mean([float(value) for value in values]) for key, values in sorted(visible_by_condition.items())},
        "fraction_multiple_visible_by_condition": {key: statistics.mean(value > 1 for value in values) for key, values in sorted(visible_by_condition.items())},
        "note": "external native selection supplies population-visible designs in every nominal condition",
    }


def input_hashes(paths: list[Path], *, anchor: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            display = path.relative_to(anchor)
        except ValueError:
            display = path
        output.append({"path": str(display), "sha256": digest, "bytes": path.stat().st_size})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO,
        help="repository or artifact root containing data/c0c3",
    )
    parser.add_argument(
        "--verify-input-hashes",
        action="store_true",
        help="fail before regenerating outputs unless the frozen raw-input hash ledger matches",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    data_root = args.data_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    campaign_root = data_root / "data/c0c3"
    greedy = campaign_root / GREEDY.name
    native = campaign_root / NATIVE.name
    fashion = campaign_root / FASHION.name
    tiny_runs = load_runs(greedy, task="tiny_adderboard", architecture="greedy", horizon=TINY_HORIZON)
    tiny_runs += load_runs(native, task="tiny_adderboard", architecture="native", horizon=TINY_HORIZON)
    fashion_runs = load_runs(fashion, task="fashion_mnist", architecture="greedy", horizon=FASHION_HORIZON)
    if len(tiny_runs) != 64:
        raise ValueError(f"Expected 64 Tiny runs, found {len(tiny_runs)}")
    if len(fashion_runs) != 20:
        raise ValueError(f"Expected 20 Fashion runs, found {len(fashion_runs)}")

    all_runs = tiny_runs + fashion_runs
    raw_paths: list[Path] = []
    for run in all_runs:
        raw_paths.extend(
            [
                run.run_dir / "manifest.json",
                run.run_dir / "state.json",
                run.run_dir / "events.jsonl",
            ]
        )
        native_path = run.run_dir / "native-openevolve/events.jsonl"
        if native_path.is_file():
            raw_paths.append(native_path)
    hashes = input_hashes(sorted(set(raw_paths)), anchor=data_root)
    hash_path = output / "input_hashes.json"
    if args.verify_input_hashes:
        if not hash_path.is_file():
            raise FileNotFoundError(f"Missing frozen input hash ledger: {hash_path}")
        expected_hashes = json.loads(hash_path.read_text(encoding="utf-8"))
        if hashes != expected_hashes:
            expected = {
                str(row["path"]): (str(row["sha256"]), int(row["bytes"]))
                for row in expected_hashes
            }
            actual = {
                str(row["path"]): (str(row["sha256"]), int(row["bytes"]))
                for row in hashes
            }
            changed = sorted(
                path
                for path in set(expected) | set(actual)
                if expected.get(path) != actual.get(path)
            )
            raise ValueError(
                "Frozen input hash verification failed for "
                f"{len(changed)} file(s): {changed[:5]}"
            )
    trajectory_rows = [trajectory_row(run) for run in all_runs]
    serializable_rows = [{key: value for key, value in row.items() if key not in {"event_details", "selected_branches"}} for row in trajectory_rows]
    write_csv(output / "trajectory_metrics.csv", serializable_rows)

    metrics = [
        "normalized_gain",
        "auc_normalized_gain",
        "final_parameters",
        "effective_top_lineages",
        "branch_delta_diversity",
        "effective_selected_parents",
        "alternative_parent_rate",
        "improvement_from_alternative_share",
        "long_lived_lineages",
        "branch_reactivations",
        "productive_branch_reactivations",
        "reference_attribution_rate",
        "broad_reference_attribution_rate",
        "comparative_evidence_rate",
        "numeric_evidence_rate",
        "tokens_per_proposal",
        "qualified_rate",
        "tail_stagnation",
    ]
    summaries = group_summaries(trajectory_rows, metrics)
    write_csv(output / "system_prompt_summaries.csv", summaries)

    contrasts: list[dict[str, Any]] = []
    for task, source in [("tiny_adderboard", trajectory_rows), ("fashion_mnist", trajectory_rows)]:
        task_rows = [row for row in source if row["task"] == task and row["architecture"] == "greedy"]
        for prompt, left, right in [("ordinary", "C0", "C2"), ("challenge", "C1", "C3")]:
            selected = [row for row in task_rows if row["condition"] in {left, right}]
            for index, metric in enumerate([
                "normalized_gain",
                "auc_normalized_gain",
                "effective_top_lineages",
                "alternative_parent_rate",
                "productive_branch_reactivations",
                "reference_attribution_rate",
                "tokens_per_proposal",
                "tail_stagnation",
            ]):
                record = paired_contrast(selected, left_condition=left, right_condition=right, metric=metric, seed=SEED + index)
                record.update({"task": task, "architecture": "greedy", "prompt_policy": prompt, "contrast": "K4_minus_K1"})
                contrasts.append(record)

    native_rows = [row for row in trajectory_rows if row["task"] == "tiny_adderboard" and row["architecture"] == "native"]
    for prompt, left, right in [("ordinary", "C0", "C2"), ("challenge", "C1", "C3")]:
        selected = [row for row in native_rows if row["condition"] in {left, right}]
        for index, metric in enumerate(["normalized_gain", "effective_top_lineages", "alternative_parent_rate", "tokens_per_proposal"]):
            record = paired_contrast(selected, left_condition=left, right_condition=right, metric=metric, seed=SEED + 100 + index)
            record.update({"task": "tiny_adderboard", "architecture": "native", "prompt_policy": prompt, "contrast": "nominal_K_label_negative_control"})
            contrasts.append(record)
    write_csv(output / "paired_contrasts.csv", contrasts)

    exemplars = trace_exemplars(trajectory_rows, all_runs)
    write_csv(output / "alternative_branch_improvements.csv", exemplars)
    validation = validate_native_semantics(tiny_runs)

    hash_path.write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    aggregate = {
        "analysis_seed": SEED,
        "tiny_common_horizon": TINY_HORIZON,
        "fashion_horizon": FASHION_HORIZON,
        "tiny_runs": len(tiny_runs),
        "fashion_runs": len(fashion_runs),
        "primary_unit": "trajectory",
        "uncertainty": "block bootstrap percentile intervals",
        "system_prompt_summaries": summaries,
        "paired_contrasts": [{key: value for key, value in row.items() if key != "differences"} for row in contrasts],
        "native_semantics_validation": validation,
        "alternative_branch_improvement_count": len(exemplars),
        "top_alternative_branch_improvements": exemplars[:20],
        "input_files_hashed": len(hashes),
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plot_system_summary(trajectory_rows, output / "fig1_system_summary.png")
    plot_lineage_raster(tiny_runs, trajectory_rows, output / "fig2_lineage_raster.png")
    plot_diversity_yield(trajectory_rows, output / "fig3_diversity_yield.png")
    plot_cross_task(contrasts, output / "fig4_cross_task.png")

    print(json.dumps({"output": str(output), "tiny_runs": len(tiny_runs), "fashion_runs": len(fashion_runs), "hashes": len(hashes)}, indent=2))


if __name__ == "__main__":
    main()
