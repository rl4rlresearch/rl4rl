#!/usr/bin/env python3
# ruff: noqa: E501
"""Reproduce Paper 4's history-refresh and search-stagnation analysis.

The analysis uses only already-recorded events from four paused semantic-v4
campaigns.  Within every campaign replicate, ``passive_control`` and
``periodic_full_refresh`` share proposals 1--5 byte-for-byte and fork at
proposal 6.  Both arms begin a fresh provider conversation every five
proposals.  The refresh arm additionally clears subject-visible outcome
history, candidate/developmental archives, and parent history while retaining
the verified incumbent model.  The estimand is therefore the whole-system
effect of periodically forgetting research history, not conversation reset.

Campaigns were paused before their planned 200-proposal endpoint.  Each
architecture-by-task stratum is analyzed only through the largest contiguous
horizon available for all six focal runs in that stratum.  No missing suffix
is imputed and no result is described as a completed 200-proposal endpoint.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import random
import re
import statistics
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_OUTPUT = HERE / "derived"
SEED = 20260901
ARMS = ("passive_control", "periodic_full_refresh")

CAMPAIGNS = (
    {
        "stratum": "Fashion / greedy",
        "task": "fashion_mnist",
        "architecture": "greedy",
        "path": REPO / "data/c0c3/semantic-interventions-v4-fashion-openevolve-campaign",
    },
    {
        "stratum": "Fashion / native",
        "task": "fashion_mnist",
        "architecture": "native",
        "path": REPO / "data/c0c3/semantic-interventions-v4-fashion-native-openevolve-campaign",
    },
    {
        "stratum": "Tiny Addition / greedy",
        "task": "tiny_adderboard",
        "architecture": "greedy",
        "path": REPO / "data/c0c3/semantic-interventions-v4-tiny-adderboard-terra-campaign",
    },
    {
        "stratum": "Tiny Addition / native",
        "task": "tiny_adderboard",
        "architecture": "native",
        "path": REPO / "data/c0c3/semantic-interventions-v4-tiny-adderboard-native-openevolve-terra-campaign",
    },
)

WORD_RE = re.compile(r"[a-z][a-z0-9_-]+", re.I)
NUMBER_RE = re.compile(r"\d")
EVIDENCE_RE = re.compile(r"\b(evidence|result|accuracy|parameter|score|failed|retained|verified)\b", re.I)
PRIOR_RE = re.compile(r"\b(prior|earlier|previous|recent|history|archive|reference design)\b", re.I)
NO_HISTORY_RE = re.compile(r"\b(no (?:earlier|prior|previous|recent) (?:evidence|result|history)|starting design|no reference design)\b", re.I)

FAMILY_PATTERNS: dict[str, re.Pattern[str]] = {
    "width_capacity": re.compile(r"\b(width|dimension|dimensional|channel|rank|hidden size|feedforward|ffn|bottleneck)\b", re.I),
    "attention_routing": re.compile(r"\b(attention|head|query|key|value|qkv|causal|routing)\b", re.I),
    "token_interface": re.compile(r"\b(token|embedding|vocab|codebook|input output|tied output)\b", re.I),
    "position": re.compile(r"\b(position|positional|relative offset|coordinate)\b", re.I),
    "normalization_bias": re.compile(r"\b(norm|normalization|bias|affine|scalar|shift|gain)\b", re.I),
    "depth_iteration": re.compile(r"\b(depth|layer|block|recurrent|iterative|shared stage|loop)\b", re.I),
    "optimization": re.compile(r"\b(optimizer|adam|learning rate|schedule|warmup|loss|batch|gradient|training step)\b", re.I),
    "convolution_spatial": re.compile(r"\b(convolution|conv|kernel|pooling|patch|spatial)\b", re.I),
    "regularization": re.compile(r"\b(dropout|weight decay|regularization|augmentation|label smoothing)\b", re.I),
}


@dataclass
class Run:
    stratum: str
    task: str
    architecture: str
    campaign: Path
    run_dir: Path
    run_id: str
    replicate: int
    arm: str
    horizon: int
    baseline: dict[str, Any]
    events: list[dict[str, Any]]
    all_rows: list[dict[str, Any]]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
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
            raise ValueError(f"Expected JSON object: {path}:{number}")
        output.append(value)
    return output


def completed_events(run_dir: Path) -> list[dict[str, Any]]:
    return sorted(
        [row for row in load_jsonl(run_dir / "events.jsonl") if row.get("event") == "proposal_completed"],
        key=lambda row: int(row["opportunity"]),
    )


def discover_runs() -> tuple[list[Run], dict[str, int]]:
    all_runs: list[Run] = []
    horizons: dict[str, int] = {}
    for spec in CAMPAIGNS:
        candidates: list[tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = []
        for run_dir in sorted((spec["path"] / "runs").iterdir()):
            manifest_path = run_dir / "manifest.json"
            events_path = run_dir / "events.jsonl"
            if not (run_dir.is_dir() and manifest_path.is_file() and events_path.is_file()):
                continue
            manifest = load_json(manifest_path)
            assignment = manifest.get("assignment") or {}
            arm = str(assignment.get("condition") or "")
            if arm not in ARMS:
                continue
            rows = load_jsonl(events_path)
            events = sorted(
                [row for row in rows if row.get("event") == "proposal_completed"],
                key=lambda row: int(row["opportunity"]),
            )
            candidates.append((run_dir, manifest, events, rows))
        if len(candidates) != 6:
            raise ValueError(f"{spec['stratum']}: expected six focal runs, found {len(candidates)}")
        horizon = min(len(value[2]) for value in candidates)
        horizons[str(spec["stratum"])] = horizon
        for run_dir, manifest, events, rows in candidates:
            assignment = manifest["assignment"]
            events = [row for row in events if int(row["opportunity"]) <= horizon]
            opportunities = [int(row["opportunity"]) for row in events]
            if opportunities != list(range(1, horizon + 1)):
                raise ValueError(f"{run_dir.name}: noncontiguous events through {horizon}")
            all_runs.append(
                Run(
                    stratum=str(spec["stratum"]),
                    task=str(spec["task"]),
                    architecture=str(spec["architecture"]),
                    campaign=spec["path"],
                    run_dir=run_dir,
                    run_id=str(assignment["run_id"]),
                    replicate=int(assignment["replicate"]),
                    arm=str(assignment["condition"]),
                    horizon=horizon,
                    baseline=manifest["baseline"],
                    events=events,
                    all_rows=rows,
                )
            )
    return all_runs, horizons


def discover_context_runs(horizons: dict[str, int]) -> list[Run]:
    all_runs: list[Run] = []
    for spec in CAMPAIGNS:
        stratum = str(spec["stratum"])
        horizon = horizons[stratum]
        for run_dir in sorted((spec["path"] / "runs").iterdir()):
            manifest_path = run_dir / "manifest.json"
            events_path = run_dir / "events.jsonl"
            if not (run_dir.is_dir() and manifest_path.is_file() and events_path.is_file()):
                continue
            manifest = load_json(manifest_path)
            assignment = manifest.get("assignment") or {}
            condition = str(assignment.get("condition") or "")
            if not condition:
                continue
            rows = load_jsonl(events_path)
            events = sorted(
                [row for row in rows if row.get("event") == "proposal_completed"],
                key=lambda row: int(row["opportunity"]),
            )
            if len(events) < horizon:
                raise ValueError(f"{run_dir.name}: context run shorter than stratum horizon {horizon}")
            events = [row for row in events if int(row["opportunity"]) <= horizon]
            all_runs.append(
                Run(
                    stratum=stratum,
                    task=str(spec["task"]),
                    architecture=str(spec["architecture"]),
                    campaign=spec["path"],
                    run_dir=run_dir,
                    run_id=str(assignment["run_id"]),
                    replicate=int(assignment["replicate"]),
                    arm=condition,
                    horizon=horizon,
                    baseline=manifest["baseline"],
                    events=events,
                    all_rows=rows,
                )
            )
    return all_runs


def validate_design(runs: list[Run], horizons: dict[str, int]) -> dict[str, Any]:
    lookup = {(run.stratum, run.replicate, run.arm): run for run in runs}
    shared_prefix_records = 0
    refresh_events = 0
    for stratum, horizon in horizons.items():
        for replicate in (1, 2, 3):
            passive = lookup[(stratum, replicate, "passive_control")]
            refresh = lookup[(stratum, replicate, "periodic_full_refresh")]
            for opportunity in range(1, 6):
                left = passive.events[opportunity - 1]
                right = refresh.events[opportunity - 1]
                if left.get("candidate_id") != right.get("candidate_id"):
                    raise ValueError(f"Prefix candidate mismatch: {stratum} r{replicate} o{opportunity}")
                if left.get("incumbent_after") != right.get("incumbent_after") or left.get("evaluation") != right.get("evaluation"):
                    raise ValueError(f"Prefix outcome mismatch: {stratum} r{replicate} o{opportunity}")
                if right.get("shared_prefix_source_run_id") != passive.run_id or not right.get("source_event_sha256"):
                    raise ValueError(f"Prefix provenance mismatch: {stratum} r{replicate} o{opportunity}")
                shared_prefix_records += 2
            expected = list(range(6, horizon + 1, 5))
            actual = sorted(
                int(row["opportunity"])
                for row in refresh.all_rows
                if row.get("event") == "search_epoch_refreshed_from_incumbent"
                and int(row.get("opportunity", 0)) <= horizon
            )
            if actual != expected:
                raise ValueError(f"Refresh ledger mismatch: {refresh.run_id}: {actual} != {expected}")
            refresh_events += len(actual)
            passive_refresh = [
                row for row in passive.all_rows
                if row.get("event") == "search_epoch_refreshed_from_incumbent"
                and int(row.get("opportunity", 0)) <= horizon
            ]
            if passive_refresh:
                raise ValueError(f"Passive arm unexpectedly refreshed: {passive.run_id}")
    return {
        "focal_trajectories": len(runs),
        "paired_replicates": len(runs) // 2,
        "logical_proposal_records": sum(len(run.events) for run in runs),
        "postfork_proposal_records": sum(max(0, len(run.events) - 5) for run in runs),
        "mirrored_prefix_records_validated": shared_prefix_records,
        "refresh_events_validated": refresh_events,
        "horizons": horizons,
    }


def validate_context(context_runs: list[Run]) -> dict[str, Any]:
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    for run in context_runs:
        strata[run.stratum][run.arm] += 1
        opportunities = [int(row["opportunity"]) for row in run.events]
        if opportunities != list(range(1, run.horizon + 1)):
            raise ValueError(f"{run.run_id}: noncontiguous context events")
    expected_conditions = {tuple(sorted(counter)) for counter in strata.values()}
    if len(expected_conditions) != 1:
        raise ValueError("Context strata do not expose the same condition set")
    for stratum, counter in strata.items():
        if any(count != 3 for count in counter.values()):
            raise ValueError(f"{stratum}: expected three runs per context condition, got {dict(counter)}")
    return {
        "context_trajectories": len(context_runs),
        "context_conditions_per_stratum": len(next(iter(strata.values()))) if strata else 0,
        "context_proposal_records": sum(len(run.events) for run in context_runs),
    }


def usage_total(value: dict[str, Any] | None) -> int:
    value = value or {}
    if value.get("total_tokens") is not None:
        return int(value.get("total_tokens") or 0)
    return int(value.get("input_tokens") or 0) + int(value.get("output_tokens") or 0)


def candidate_table(run: Run) -> dict[str, dict[str, Any]]:
    root = run.baseline
    table: dict[str, dict[str, Any]] = {
        str(root["candidate_id"]): {
            "candidate_id": str(root["candidate_id"]),
            "metrics": root.get("metrics") or {},
            "fitness": root.get("fitness"),
            "created_opportunity": 0,
        }
    }
    for event in run.events:
        candidate_id = str(event.get("candidate_id") or "")
        if not candidate_id:
            continue
        evaluation = event.get("evaluation") or {}
        table[candidate_id] = {
            "candidate_id": candidate_id,
            "metrics": evaluation.get("metrics") or {},
            "fitness": evaluation.get("fitness"),
            "created_opportunity": int(event["opportunity"]),
        }
    return table


def candidate_objective(run: Run, candidate_id: str, table: dict[str, dict[str, Any]]) -> float:
    value = table.get(candidate_id) or {}
    metrics = value.get("metrics") or {}
    if run.task == "tiny_adderboard":
        parameters = metrics.get("parameters")
        return -float(parameters) if parameters is not None else math.nan
    correct = metrics.get("validation_correct")
    if correct is not None:
        return float(correct)
    fitness = value.get("fitness")
    return float(fitness) if fitness is not None else math.nan


def source_path(run: Run, candidate_id: str) -> Path:
    return run.run_dir / "candidates" / candidate_id / "train.py"


def proposal_source_path(run: Run, event: dict[str, Any]) -> Path:
    candidate = source_path(run, str(event.get("candidate_id") or ""))
    if candidate.is_file():
        return candidate
    opportunity = int(event["opportunity"])
    return run.run_dir / "opportunities" / f"{opportunity:04d}" / "evaluation-workspace" / "train.py"


def normalized_python_ngrams(path: Path, n: int = 3) -> set[tuple[str, ...]]:
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    values: list[str] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in stream:
            if token.type in {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT}:
                continue
            if token.type == tokenize.NAME:
                if token.string in {"def", "class", "return", "if", "else", "for", "while", "in", "is", "not", "and", "or", "lambda", "with", "as", "import", "from", "raise", "try", "except", "finally", "True", "False", "None"}:
                    values.append(token.string)
                else:
                    values.append("NAME")
            elif token.type == tokenize.NUMBER:
                values.append("NUMBER")
            elif token.type == tokenize.STRING:
                values.append("STRING")
            else:
                values.append(token.string)
    except (tokenize.TokenError, IndentationError):
        return set()
    return {tuple(values[index:index + n]) for index in range(max(0, len(values) - n + 1))}


def ast_multiset(path: Path) -> Counter[str]:
    if not path.is_file():
        return Counter()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return Counter()
    return Counter(type(node).__name__ for node in ast.walk(tree))


def jaccard_distance(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return 1.0 - len(left & right) / len(union) if union else 0.0


def ast_distance(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    denominator = sum(max(left[key], right[key]) for key in keys)
    numerator = sum(abs(left[key] - right[key]) for key in keys)
    return numerator / denominator if denominator else 0.0


def words(text: str) -> set[str]:
    return {value.lower() for value in WORD_RE.findall(text)}


def lexical_novelty(text: str, prior: list[str]) -> float:
    current = words(text)
    if not prior:
        return 1.0 if current else 0.0
    return min(jaccard_distance(current, words(value)) for value in prior)


def normalize_phrase(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def mechanism_families(text: str) -> set[str]:
    return {name for name, pattern in FAMILY_PATTERNS.items() if pattern.search(text)}


def message_metadata(run: Run, opportunity: int, event: dict[str, Any]) -> tuple[str, str, str, str]:
    """Parse the saved subject message, bypassing a native-adapter field bug."""
    path = run.run_dir / "opportunities" / f"{opportunity:04d}" / "codex" / f"proposal-{opportunity}.last-message.md"
    values = {
        "MECHANISM": str(event.get("mechanism") or ""),
        "HYPOTHESIS": str(event.get("hypothesis") or ""),
        "INTENDED_EDIT": str(event.get("intended_edit") or ""),
        "EVIDENCE": str(event.get("evidence") or ""),
    }
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="replace")
        for key in tuple(values):
            match = re.search(rf"(?mi)^{key}:\s*(.+?)\s*$", text)
            if match:
                values[key] = match.group(1).strip()
    return values["MECHANISM"], values["HYPOTHESIS"], values["INTENDED_EDIT"], values["EVIDENCE"]


def event_and_trajectory_rows(run: Run) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = candidate_table(run)
    event_rows: list[dict[str, Any]] = []
    mechanism_history: list[str] = []
    hypothesis_history: list[str] = []
    family_history: set[str] = set()
    source_history: list[set[tuple[str, ...]]] = []
    incumbent_before = str(run.baseline["candidate_id"])
    fork_candidate = ""
    fork_objective = math.nan
    progress_series: list[float] = []
    retained_opportunities: list[int] = []
    drought = 0
    max_drought = 0
    previous_progress = 0.0

    for event in run.events:
        opportunity = int(event["opportunity"])
        candidate_id = str(event.get("candidate_id") or "")
        parent_ids = event.get("selected_parent_ids") or event.get("parent_ids") or []
        parent_id = str(parent_ids[0]) if parent_ids else incumbent_before
        incumbent_after = str(event.get("incumbent_after") or incumbent_before)
        if opportunity == 5:
            fork_candidate = incumbent_after
            fork_objective = candidate_objective(run, fork_candidate, table)
        candidate_source_file = proposal_source_path(run, event)
        candidate_source = normalized_python_ngrams(candidate_source_file)
        parent_source = normalized_python_ngrams(source_path(run, parent_id))
        source_novelty = jaccard_distance(candidate_source, parent_source) if candidate_source and parent_source else math.nan
        prior_source_novelty = (
            min(jaccard_distance(candidate_source, value) for value in source_history)
            if candidate_source and source_history else 1.0
        )
        candidate_ast = ast_multiset(candidate_source_file)
        parent_ast = ast_multiset(source_path(run, parent_id))
        ast_delta = ast_distance(candidate_ast, parent_ast) if candidate_ast and parent_ast else math.nan
        mechanism, hypothesis, intended_edit, evidence = message_metadata(run, opportunity, event)
        declared = " ".join((mechanism, hypothesis, intended_edit))
        families = mechanism_families(declared)
        new_family = bool(families - family_history)
        repeated_mechanism = bool(normalize_phrase(mechanism) and normalize_phrase(mechanism) in {normalize_phrase(value) for value in mechanism_history})
        evaluation = event.get("evaluation") or {}
        metrics = evaluation.get("metrics") or {}
        if run.task == "tiny_adderboard":
            qualified = bool(evaluation.get("valid")) and float(metrics.get("accuracy") or 0.0) >= 0.99
        else:
            qualified = bool(evaluation.get("valid"))
        retained = bool(event.get("retained"))
        if retained:
            retained_opportunities.append(opportunity)
            drought = 0
        else:
            drought += 1
            max_drought = max(max_drought, drought)
        current_objective = candidate_objective(run, incumbent_after, table)
        if opportunity >= 5 and math.isfinite(fork_objective) and math.isfinite(current_objective):
            if run.task == "tiny_adderboard":
                progress = (current_objective - fork_objective) / max(1.0, -fork_objective)
            else:
                progress = (current_objective - fork_objective) / max(1.0, 10000.0 - fork_objective)
            if opportunity > 5:
                progress_series.append(progress)
        else:
            progress = math.nan
        progress_increment = (
            progress - previous_progress
            if opportunity > 5 and math.isfinite(progress)
            else 0.0
        )
        if opportunity >= 5 and math.isfinite(progress):
            previous_progress = progress
        phase_start = opportunity >= 6 and (opportunity - 6) % 5 == 0
        event_rows.append(
            {
                "stratum": run.stratum,
                "task": run.task,
                "architecture": run.architecture,
                "replicate": run.replicate,
                "arm": run.arm,
                "run_id": run.run_id,
                "horizon": run.horizon,
                "opportunity": opportunity,
                "postfork": int(opportunity >= 6),
                "phase": max(0, (opportunity - 1) // 5),
                "phase_start": int(phase_start),
                "retained": int(retained),
                "qualified": int(qualified),
                "valid": int(bool(evaluation.get("valid"))),
                "source_novelty": source_novelty,
                "source_novelty_to_prior_min": prior_source_novelty,
                "candidate_source_available": int(bool(candidate_source)),
                "parent_source_available": int(bool(parent_source)),
                "ast_distance": ast_delta,
                "mechanism_lexical_novelty": lexical_novelty(mechanism, mechanism_history),
                "hypothesis_lexical_novelty": lexical_novelty(hypothesis, hypothesis_history),
                "exact_mechanism_repeat": int(repeated_mechanism),
                "new_mechanism_family": int(new_family),
                "mechanism_family_count": len(families),
                "mechanism_families": ";".join(sorted(families)),
                "evidence_language": int(bool(EVIDENCE_RE.search(evidence))),
                "prior_history_language": int(bool(PRIOR_RE.search(evidence + " " + hypothesis))),
                "no_history_language": int(bool(NO_HISTORY_RE.search(evidence + " " + hypothesis))),
                "numeric_evidence": int(bool(NUMBER_RE.search(evidence))),
                "mechanism": mechanism,
                "hypothesis": hypothesis,
                "evidence": evidence,
                "candidate_id": candidate_id,
                "parent_id": parent_id,
                "incumbent_after": incumbent_after,
                "incumbent_objective": current_objective,
                "normalized_progress_from_fork": progress,
                "progress_increment": progress_increment,
                "tokens": usage_total(event.get("usage_increment")),
                "input_tokens": int((event.get("usage_increment") or {}).get("input_tokens") or 0),
                "cached_input_tokens": int((event.get("usage_increment") or {}).get("cached_input_tokens") or 0),
                "output_tokens": int((event.get("usage_increment") or {}).get("output_tokens") or 0),
                "evaluator_seconds": float(event.get("evaluator_seconds_increment") or 0.0),
            }
        )
        if candidate_source:
            source_history.append(candidate_source)
        mechanism_history.append(mechanism)
        hypothesis_history.append(hypothesis)
        family_history |= families
        incumbent_before = incumbent_after

    if not fork_candidate or not math.isfinite(fork_objective):
        raise ValueError(f"Missing fork state: {run.run_id}")
    final_candidate = str(run.events[-1].get("incumbent_after") or "")
    final_objective = candidate_objective(run, final_candidate, table)
    post = [row for row in event_rows if row["postfork"]]
    if run.task == "tiny_adderboard":
        fork_parameters = -fork_objective
        final_parameters = -final_objective
        endpoint_progress = (fork_parameters - final_parameters) / max(1.0, fork_parameters)
        endpoint_value = final_parameters
    else:
        fork_parameters = math.nan
        final_parameters = math.nan
        endpoint_progress = (final_objective - fork_objective) / max(1.0, 10000.0 - fork_objective)
        endpoint_value = final_objective
    last_improvement = max(retained_opportunities, default=5)
    post_improvements = [value for value in retained_opportunities if value >= 6]
    row = {
        "stratum": run.stratum,
        "task": run.task,
        "architecture": run.architecture,
        "replicate": run.replicate,
        "arm": run.arm,
        "run_id": run.run_id,
        "horizon": run.horizon,
        "postfork_proposals": len(post),
        "fork_objective": fork_objective,
        "endpoint_value": endpoint_value,
        "endpoint_progress": endpoint_progress,
        "auc_progress": statistics.mean(progress_series) if progress_series else 0.0,
        "fork_parameters": fork_parameters,
        "final_parameters": final_parameters,
        "retained_count": sum(int(value["retained"]) for value in post),
        "retained_rate": statistics.mean(int(value["retained"]) for value in post),
        "qualified_rate": statistics.mean(int(value["qualified"]) for value in post),
        "last_improvement": last_improvement,
        "tail_drought": run.horizon - last_improvement,
        "max_drought": max_drought,
        "phase_start_retained_rate": statistics.mean(int(value["retained"]) for value in post if value["phase_start"]),
        "within_phase_retained_rate": statistics.mean(int(value["retained"]) for value in post if not value["phase_start"]),
        "mean_source_novelty": mean([float(value["source_novelty"]) for value in post]),
        "mean_prior_source_novelty": mean([float(value["source_novelty_to_prior_min"]) for value in post]),
        "candidate_source_available_rate": statistics.mean(int(value["candidate_source_available"]) for value in post),
        "parent_source_available_rate": statistics.mean(int(value["parent_source_available"]) for value in post),
        "mean_ast_distance": mean([float(value["ast_distance"]) for value in post]),
        "mean_mechanism_lexical_novelty": mean([float(value["mechanism_lexical_novelty"]) for value in post]),
        "exact_mechanism_repeat_rate": statistics.mean(int(value["exact_mechanism_repeat"]) for value in post),
        "new_family_rate": statistics.mean(int(value["new_mechanism_family"]) for value in post),
        "prior_history_language_rate": statistics.mean(int(value["prior_history_language"]) for value in post),
        "no_history_language_rate": statistics.mean(int(value["no_history_language"]) for value in post),
        "numeric_evidence_rate": statistics.mean(int(value["numeric_evidence"]) for value in post),
        "tokens": sum(int(value["tokens"]) for value in post),
        "input_tokens": sum(int(value["input_tokens"]) for value in post),
        "cached_input_tokens": sum(int(value["cached_input_tokens"]) for value in post),
        "output_tokens": sum(int(value["output_tokens"]) for value in post),
        "evaluator_seconds": sum(float(value["evaluator_seconds"]) for value in post),
        "tokens_per_retention": sum(int(value["tokens"]) for value in post) / len(post_improvements) if post_improvements else math.nan,
        "progress_per_retention": endpoint_progress / len(post_improvements) if post_improvements else math.nan,
        "largest_progress_jump": max(float(value["progress_increment"]) for value in post),
    }
    return event_rows, row


def context_trajectory_row(run: Run) -> dict[str, Any]:
    table = candidate_table(run)
    incumbent_before = str(run.baseline["candidate_id"])
    fork_objective = math.nan
    progress_series: list[float] = []
    post_rows: list[dict[str, Any]] = []
    for event in run.events:
        opportunity = int(event["opportunity"])
        incumbent_after = str(event.get("incumbent_after") or incumbent_before)
        if opportunity == 5:
            fork_objective = candidate_objective(run, incumbent_after, table)
        current_objective = candidate_objective(run, incumbent_after, table)
        if opportunity >= 5 and math.isfinite(fork_objective) and math.isfinite(current_objective):
            if run.task == "tiny_adderboard":
                progress = (current_objective - fork_objective) / max(1.0, -fork_objective)
            else:
                progress = (current_objective - fork_objective) / max(1.0, 10000.0 - fork_objective)
            if opportunity > 5:
                progress_series.append(progress)
        else:
            progress = math.nan
        if opportunity >= 6:
            post_rows.append(
                {
                    "retained": int(bool(event.get("retained"))),
                    "tokens": usage_total(event.get("usage_increment")),
                    "progress": progress,
                }
            )
        incumbent_before = incumbent_after
    final_candidate = str(run.events[-1].get("incumbent_after") or "")
    final_objective = candidate_objective(run, final_candidate, table)
    if run.task == "tiny_adderboard":
        endpoint_value = -final_objective
        endpoint_progress = (-fork_objective - endpoint_value) / max(1.0, -fork_objective)
    else:
        endpoint_value = final_objective
        endpoint_progress = (final_objective - fork_objective) / max(1.0, 10000.0 - fork_objective)
    return {
        "stratum": run.stratum,
        "task": run.task,
        "architecture": run.architecture,
        "replicate": run.replicate,
        "condition": run.arm,
        "run_id": run.run_id,
        "horizon": run.horizon,
        "postfork_proposals": len(post_rows),
        "endpoint_value": endpoint_value,
        "endpoint_progress": endpoint_progress,
        "auc_progress": statistics.mean(progress_series) if progress_series else 0.0,
        "retained_rate": statistics.mean(row["retained"] for row in post_rows),
        "tokens": sum(int(row["tokens"]) for row in post_rows),
    }


def condition_context_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["stratum"]), str(row["condition"]))].append(row)
    output: list[dict[str, Any]] = []
    for (stratum, condition), values in sorted(grouped.items()):
        output.append(
            {
                "stratum": stratum,
                "condition": condition,
                "runs": len(values),
                "horizon": values[0]["horizon"],
                "endpoint_progress_mean": mean([float(row["endpoint_progress"]) for row in values]),
                "auc_progress_mean": mean([float(row["auc_progress"]) for row in values]),
                "retained_rate_mean": mean([float(row["retained_rate"]) for row in values]),
                "tokens_mean": mean([float(row["tokens"]) for row in values]),
            }
        )
    for stratum in sorted({row["stratum"] for row in output}):
        subset = [row for row in output if row["stratum"] == stratum]
        endpoint_sorted = sorted(subset, key=lambda row: (-float(row["endpoint_progress_mean"]), str(row["condition"])))
        token_sorted = sorted(subset, key=lambda row: (float(row["tokens_mean"]), str(row["condition"])))
        for rank, row in enumerate(endpoint_sorted, 1):
            row["endpoint_rank"] = rank
        for rank, row in enumerate(token_sorted, 1):
            row["token_rank"] = rank
    return sorted(output, key=lambda row: (row["stratum"], int(row["endpoint_rank"]), row["condition"]))


def context_summary(condition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    refresh_rows = [row for row in condition_rows if row["condition"] == "periodic_full_refresh"]
    passive_rows = [row for row in condition_rows if row["condition"] == "passive_control"]
    output = {
        "periodic_full_refresh_by_stratum": refresh_rows,
        "passive_control_by_stratum": passive_rows,
        "refresh_top_third_endpoint_strata": sum(int(row["endpoint_rank"]) <= 8 for row in refresh_rows),
        "refresh_lowest_half_tokens_strata": sum(int(row["token_rank"]) <= 12 for row in refresh_rows),
    }
    return output


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.mean(finite) if finite else math.nan


def paired_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    lookup = {(row["stratum"], int(row["replicate"]), row["arm"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for spec in CAMPAIGNS:
        stratum = str(spec["stratum"])
        for replicate in (1, 2, 3):
            passive = lookup[(stratum, replicate, "passive_control")]
            refresh = lookup[(stratum, replicate, "periodic_full_refresh")]
            left = float(passive[metric])
            right = float(refresh[metric])
            output.append(
                {
                    "stratum": stratum,
                    "task": passive["task"],
                    "architecture": passive["architecture"],
                    "replicate": replicate,
                    "metric": metric,
                    "passive": left,
                    "refresh": right,
                    "refresh_minus_passive": right - left,
                }
            )
    return output


def stratified_bootstrap(pairs: list[dict[str, Any]], *, seed: int, samples: int = 20000) -> tuple[float, float]:
    by_stratum: dict[str, list[float]] = defaultdict(list)
    for row in pairs:
        value = float(row["refresh_minus_passive"])
        if math.isfinite(value):
            by_stratum[str(row["stratum"])].append(value)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        stratum_means = [statistics.mean(rng.choices(items, k=len(items))) for items in by_stratum.values()]
        values.append(statistics.mean(stratum_means))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def contrast_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "endpoint_progress", "auc_progress", "retained_rate", "qualified_rate",
        "tail_drought", "max_drought", "phase_start_retained_rate", "within_phase_retained_rate",
        "mean_source_novelty", "mean_prior_source_novelty", "mean_ast_distance",
        "mean_mechanism_lexical_novelty", "exact_mechanism_repeat_rate", "new_family_rate",
        "candidate_source_available_rate", "parent_source_available_rate",
        "prior_history_language_rate", "no_history_language_rate", "numeric_evidence_rate",
        "tokens", "input_tokens", "cached_input_tokens", "output_tokens", "evaluator_seconds", "tokens_per_retention",
        "progress_per_retention", "largest_progress_jump",
    )
    output: list[dict[str, Any]] = []
    for index, metric in enumerate(metrics):
        pairs = paired_rows(rows, metric)
        finite = [row for row in pairs if math.isfinite(float(row["refresh_minus_passive"]))]
        differences = [float(row["refresh_minus_passive"]) for row in finite]
        low, high = stratified_bootstrap(finite, seed=SEED + index) if finite else (math.nan, math.nan)
        output.append(
            {
                "metric": metric,
                "n_pairs": len(finite),
                "passive_mean": mean([float(row["passive"]) for row in finite]),
                "refresh_mean": mean([float(row["refresh"]) for row in finite]),
                "paired_difference": mean(differences),
                "bootstrap_low": low,
                "bootstrap_high": high,
                "refresh_higher": sum(value > 0 for value in differences),
                "passive_higher": sum(value < 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
            }
        )
    return output


def stratum_contrast_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "endpoint_progress",
        "auc_progress",
        "retained_rate",
        "phase_start_retained_rate",
        "within_phase_retained_rate",
        "mean_mechanism_lexical_novelty",
        "prior_history_language_rate",
        "tokens",
    )
    output: list[dict[str, Any]] = []
    for metric in metrics:
        for stratum in [str(spec["stratum"]) for spec in CAMPAIGNS]:
            pairs = [row for row in paired_rows(rows, metric) if row["stratum"] == stratum and math.isfinite(float(row["refresh_minus_passive"]))]
            differences = [float(row["refresh_minus_passive"]) for row in pairs]
            output.append(
                {
                    "metric": metric,
                    "stratum": stratum,
                    "n_pairs": len(pairs),
                    "passive_mean": mean([float(row["passive"]) for row in pairs]),
                    "refresh_mean": mean([float(row["refresh"]) for row in pairs]),
                    "paired_difference": mean(differences),
                    "refresh_higher": sum(value > 0 for value in differences),
                    "passive_higher": sum(value < 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                }
            )
    return output


def leave_one_stratum_out_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("endpoint_progress", "auc_progress", "retained_rate", "phase_start_retained_rate", "tokens")
    strata = [str(spec["stratum"]) for spec in CAMPAIGNS]
    output: list[dict[str, Any]] = []
    for metric in metrics:
        pairs = [row for row in paired_rows(rows, metric) if math.isfinite(float(row["refresh_minus_passive"]))]
        for omitted in strata:
            kept = [row for row in pairs if row["stratum"] != omitted]
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in kept:
                grouped[str(row["stratum"])].append(float(row["refresh_minus_passive"]))
            stratum_means = [statistics.mean(values) for values in grouped.values()]
            differences = [float(row["refresh_minus_passive"]) for row in kept]
            output.append(
                {
                    "metric": metric,
                    "omitted_stratum": omitted,
                    "remaining_pairs": len(kept),
                    "paired_difference_equal_strata": statistics.mean(stratum_means) if stratum_means else math.nan,
                    "refresh_higher": sum(value > 0 for value in differences),
                    "passive_higher": sum(value < 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                }
            )
    return output


def phase_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        if not row["postfork"]:
            continue
        grouped[(str(row["stratum"]), int(row["replicate"]), str(row["arm"]), int(row["phase"]))].append(row)
    output: list[dict[str, Any]] = []
    for (stratum, replicate, arm, phase), values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["opportunity"]))
        output.append(
            {
                "stratum": stratum,
                "replicate": replicate,
                "arm": arm,
                "phase": phase,
                "start_opportunity": min(int(row["opportunity"]) for row in values),
                "n_proposals": len(values),
                "any_retained": int(any(row["retained"] for row in values)),
                "retained_count": sum(int(row["retained"]) for row in values),
                "first_retained": int(values[0]["retained"]),
                "mean_source_novelty": mean([float(row["source_novelty"]) for row in values]),
                "mean_mechanism_novelty": mean([float(row["mechanism_lexical_novelty"]) for row in values]),
                "mechanism_repeat_rate": statistics.mean(int(row["exact_mechanism_repeat"]) for row in values),
                "tokens": sum(int(row["tokens"]) for row in values),
                "progress_end": float(values[-1]["normalized_progress_from_fork"]),
            }
        )
    return output


def phase_start_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ARMS:
        values = [
            row for row in events
            if row["postfork"] and row["phase_start"] and row["arm"] == arm
        ]
        family_counts: Counter[str] = Counter()
        for row in values:
            family_counts.update(value for value in str(row["mechanism_families"]).split(";") if value)
        output[arm] = {
            "phase_start_events": len(values),
            "retained": sum(int(row["retained"]) for row in values),
            "retained_rate": statistics.mean(int(row["retained"]) for row in values),
            "prior_history_language": sum(int(row["prior_history_language"]) for row in values),
            "prior_history_language_rate": statistics.mean(int(row["prior_history_language"]) for row in values),
            "numeric_evidence": sum(int(row["numeric_evidence"]) for row in values),
            "numeric_evidence_rate": statistics.mean(int(row["numeric_evidence"]) for row in values),
            "mechanism_family_counts": dict(sorted(family_counts.items())),
        }
    return output


def qualitative_examples(events: list[dict[str, Any]], trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    pair_progress = paired_rows(trajectories, "endpoint_progress")
    most_helped = max(pair_progress, key=lambda row: float(row["refresh_minus_passive"]))
    most_hurt = min(pair_progress, key=lambda row: float(row["refresh_minus_passive"]))
    anchors: dict[str, Any] = {"most_helped_pair": most_helped, "most_hurt_pair": most_hurt, "phase_start_examples": []}
    for label, pair in (("helped", most_helped), ("hurt", most_hurt)):
        for arm in ARMS:
            candidates = [
                row for row in events
                if row["stratum"] == pair["stratum"]
                and int(row["replicate"]) == int(pair["replicate"])
                and row["arm"] == arm
                and row["phase_start"]
            ]
            retained = [row for row in candidates if row["retained"]]
            chosen = (retained or candidates)[:3]
            for row in chosen:
                anchors["phase_start_examples"].append(
                    {
                        "case": label,
                        "stratum": row["stratum"],
                        "replicate": row["replicate"],
                        "arm": arm,
                        "opportunity": row["opportunity"],
                        "retained": row["retained"],
                        "mechanism": row["mechanism"],
                        "hypothesis": row["hypothesis"],
                        "evidence": row["evidence"],
                        "run_id": row["run_id"],
                    }
                )
    return anchors


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_progress(events: list[dict[str, Any]], output: Path) -> None:
    strata = [str(spec["stratum"]) for spec in CAMPAIGNS]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1), sharey=True)
    colors = {"passive_control": "#59636f", "periodic_full_refresh": "#c9485b"}
    labels = {"passive_control": "History retained", "periodic_full_refresh": "History refreshed"}
    for axis, stratum in zip(axes.flat, strata):
        subset = [row for row in events if row["stratum"] == stratum and row["opportunity"] >= 5]
        for arm in ARMS:
            arm_rows = [row for row in subset if row["arm"] == arm]
            opportunities = sorted({int(row["opportunity"]) for row in arm_rows})
            means = [mean([float(row["normalized_progress_from_fork"]) for row in arm_rows if int(row["opportunity"]) == opportunity]) for opportunity in opportunities]
            axis.plot(opportunities, means, color=colors[arm], lw=1.7, label=labels[arm])
            for replicate in (1, 2, 3):
                values = [row for row in arm_rows if int(row["replicate"]) == replicate]
                axis.plot([int(row["opportunity"]) for row in values], [float(row["normalized_progress_from_fork"]) for row in values], color=colors[arm], alpha=0.18, lw=0.7)
        for opportunity in range(6, max(int(row["opportunity"]) for row in subset) + 1, 5):
            axis.axvline(opportunity, color="#d9dde1", lw=0.45, zorder=0)
        axis.axhline(0, color="#79818a", lw=0.6)
        axis.set_title(stratum, fontsize=9)
        axis.set_xlabel("Proposal")
        axis.set_ylabel("Progress from shared fork")
        axis.grid(axis="y", color="#eceef0", lw=0.5)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "figure1_progress.pdf", bbox_inches="tight")
    fig.savefig(output / "figure1_progress.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def box(axis: Any, xy: tuple[float, float], width: float, height: float, text: str, *, face: str, edge: str = "#334155") -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8.2, color="#111827")


def plot_design(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.1, 2.65))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.03, 0.93, "State-matched history-refresh design", fontsize=10, weight="bold", color="#111827")
    axis.text(0.03, 0.84, "Same evaluator, objective, incumbent, and conversation-reset cadence; different visible history.", fontsize=8.1, color="#475569")
    box(axis, (0.05, 0.49), 0.24, 0.18, "Proposals 1-5\nbyte-mirrored", face="#eef2ff", edge="#6366f1")
    box(axis, (0.39, 0.64), 0.25, 0.15, "Passive control\nkeeps archive + path", face="#f1f5f9")
    box(axis, (0.39, 0.33), 0.25, 0.15, "Periodic refresh\nkeeps incumbent only", face="#fff1f2", edge="#e11d48")
    box(axis, (0.73, 0.64), 0.19, 0.15, "Continue from\nsame incumbent", face="#f8fafc")
    box(axis, (0.73, 0.33), 0.19, 0.15, "Continue from\nsame incumbent", face="#fff7ed", edge="#f97316")
    arrowprops = {"arrowstyle": "->", "color": "#64748b", "lw": 1.2}
    axis.annotate("", xy=(0.39, 0.715), xytext=(0.29, 0.58), arrowprops=arrowprops)
    axis.annotate("", xy=(0.39, 0.405), xytext=(0.29, 0.58), arrowprops=arrowprops)
    axis.annotate("", xy=(0.73, 0.715), xytext=(0.64, 0.715), arrowprops=arrowprops)
    axis.annotate("", xy=(0.73, 0.405), xytext=(0.64, 0.405), arrowprops=arrowprops)
    axis.text(0.405, 0.24, "Refresh opportunities: 6, 11, 16, ...", fontsize=7.8, color="#be123c")
    axis.text(0.405, 0.16, "Not a task restart: the incumbent program is preserved.", fontsize=7.8, color="#334155")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output / "figure0_design.pdf", bbox_inches="tight")
    fig.savefig(output / "figure0_design.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_process(trajectories: list[dict[str, Any]], output: Path) -> None:
    metrics = [
        ("retained_rate", "Retention rate"),
        ("mean_prior_source_novelty", "Novelty vs any prior source"),
        ("exact_mechanism_repeat_rate", "Exact mechanism-repeat rate"),
        ("prior_history_language_rate", "Prior-history language rate"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    colors = {"passive_control": "#59636f", "periodic_full_refresh": "#c9485b"}
    for axis, (metric, title) in zip(axes, metrics):
        pairs = paired_rows(trajectories, metric)
        for index, row in enumerate(pairs):
            axis.plot([0, 1], [row["passive"], row["refresh"]], color="#c7ccd1", lw=0.7, alpha=0.8)
            axis.scatter([0], [row["passive"]], color=colors["passive_control"], s=12, zorder=3)
            axis.scatter([1], [row["refresh"]], color=colors["periodic_full_refresh"], s=12, zorder=3)
        axis.set_xticks([0, 1], ["Retain", "Refresh"], fontsize=7)
        axis.set_title(title, fontsize=8)
        axis.grid(axis="y", color="#eceef0", lw=0.5)
    fig.tight_layout()
    fig.savefig(output / "figure2_process.pdf", bbox_inches="tight")
    fig.savefig(output / "figure2_process.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def input_files(runs: list[Run], context_runs: list[Run]) -> list[Path]:
    paths: set[Path] = set()
    for run in context_runs:
        paths.update({run.campaign / "campaign.json", run.campaign / "semantic-interventions.json", run.run_dir / "manifest.json", run.run_dir / "events.jsonl"})
    for run in runs:
        paths.update({run.campaign / "campaign.json", run.campaign / "semantic-interventions.json", run.run_dir / "manifest.json", run.run_dir / "events.jsonl"})
        for event in run.events:
            opportunity = int(event["opportunity"])
            proposal_source = proposal_source_path(run, event)
            if proposal_source.is_file():
                paths.add(proposal_source)
            for candidate_id in [str(event.get("candidate_id") or ""), *(str(value) for value in (event.get("selected_parent_ids") or event.get("parent_ids") or []))]:
                path = source_path(run, candidate_id)
                if candidate_id and path.is_file():
                    paths.add(path)
            message = run.run_dir / "opportunities" / f"{opportunity:04d}" / "codex" / f"proposal-{opportunity}.last-message.md"
            if message.is_file():
                paths.add(message)
    return sorted(paths)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(output: Path, files: list[Path], data_root: Path) -> None:
    ledger_path = output / "input_hashes.json"
    if not ledger_path.is_file():
        raise ValueError(f"Hash ledger unavailable: {ledger_path}")
    expected = load_json(ledger_path).get("files") or {}
    actual = {str(path.relative_to(data_root)): sha256(path) for path in files}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))[:5]
        added = sorted(set(actual) - set(expected))[:5]
        changed = sorted(key for key in set(expected) & set(actual) if expected[key] != actual[key])[:5]
        raise ValueError(f"Input hash mismatch; missing={missing}, added={added}, changed={changed}")


def main() -> None:
    global REPO, CAMPAIGNS
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-input-hashes", action="store_true")
    args = parser.parse_args()
    if args.data_root.resolve() != REPO.resolve():
        root = args.data_root.resolve()
        REPO = root
        CAMPAIGNS = tuple({**spec, "path": root / spec["path"].relative_to(HERE.parents[2])} for spec in CAMPAIGNS)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs, horizons = discover_runs()
    context_runs = discover_context_runs(horizons)
    design = validate_design(runs, horizons)
    context_design = validate_context(context_runs)
    files = input_files(runs, context_runs)
    if args.verify_input_hashes:
        verify_hashes(output, files, REPO)
    all_events: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for run in runs:
        event_rows, trajectory = event_and_trajectory_rows(run)
        all_events.extend(event_rows)
        trajectories.append(trajectory)
    phases = phase_rows(all_events)
    phase_start = phase_start_summary(all_events)
    context_trajectories = [context_trajectory_row(run) for run in context_runs]
    condition_context = condition_context_rows(context_trajectories)
    context = context_summary(condition_context)
    contrasts = contrast_table(trajectories)
    by_stratum = stratum_contrast_rows(trajectories)
    leave_one_out = leave_one_stratum_out_rows(trajectories)
    qualitative = qualitative_examples(all_events, trajectories)
    write_csv(output / "event_metrics.csv", all_events)
    write_csv(output / "trajectory_metrics.csv", trajectories)
    write_csv(output / "phase_metrics.csv", phases)
    write_csv(output / "paired_contrasts.csv", contrasts)
    write_csv(output / "stratum_contrasts.csv", by_stratum)
    write_csv(output / "leave_one_stratum_out.csv", leave_one_out)
    write_csv(output / "context_trajectory_metrics.csv", context_trajectories)
    write_csv(output / "condition_context.csv", condition_context)
    (output / "phase_start_summary.json").write_text(json.dumps(phase_start, indent=2, sort_keys=True), encoding="utf-8")
    (output / "context_summary.json").write_text(json.dumps(context, indent=2, sort_keys=True), encoding="utf-8")
    pair_details: list[dict[str, Any]] = []
    for metric in ("endpoint_progress", "auc_progress", "retained_rate", "tail_drought", "mean_prior_source_novelty", "exact_mechanism_repeat_rate", "tokens"):
        pair_details.extend(paired_rows(trajectories, metric))
    write_csv(output / "pair_details.csv", pair_details)
    (output / "qualitative_examples.json").write_text(json.dumps(qualitative, indent=2, sort_keys=True), encoding="utf-8")
    hashes = {str(path.relative_to(REPO)): sha256(path) for path in files}
    (output / "input_hashes.json").write_text(json.dumps({"algorithm": "sha256", "file_count": len(hashes), "files": hashes}, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "analysis_seed": SEED,
        "design": design,
        "context_design": context_design,
        "context_summary": context,
        "contrast_definition": "periodic_full_refresh minus passive_control; equal weight to each architecture-by-task stratum in pooled descriptive contrasts",
        "contrasts": {row["metric"]: row for row in contrasts},
        "trajectory_count": len(trajectories),
        "event_count": len(all_events),
        "input_file_count": len(hashes),
        "phase_start_summary": phase_start,
        "robustness": {
            "endpoint_leave_one_stratum_out": [row for row in leave_one_out if row["metric"] == "endpoint_progress"],
            "auc_leave_one_stratum_out": [row for row in leave_one_out if row["metric"] == "auc_progress"],
            "endpoint_by_stratum": [row for row in by_stratum if row["metric"] == "endpoint_progress"],
        },
    }
    (output / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    plot_design(output)
    plot_progress(all_events, output)
    plot_process(trajectories, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
