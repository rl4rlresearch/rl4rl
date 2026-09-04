#!/usr/bin/env python3
# ruff: noqa: E501
"""Reproduce Paper 5's interface-induced behavior analysis.

This script uses already-recorded autonomous-research traces.  It treats the
research interface itself as the object of study: continuous Autoresearch
sessions, bounded greedy OpenEvolve-style patch calls, and native
OpenEvolve-style population calls.

The comparisons are deliberately descriptive.  They are task- and
interface-composed contrasts, not randomized estimates of a latent model
capability.  Fashion-MNIST supplies the main same-task contrast because both
Autoresearch and bounded greedy OpenEvolve used the same seed candidate and
task surface.  NanoGPT supplies an early-horizon stress check.  Tiny Addition
supplies a same-protocol greedy-vs-native OpenEvolve interface check.
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_OUTPUT = HERE / "derived"
SEED = 20260902


@dataclass(frozen=True)
class CampaignSpec:
    key: str
    task: str
    comparison: str
    interface: str
    framework_id: str
    path: Path
    blocks: tuple[int, ...]
    horizon: int


CAMPAIGNS = (
    CampaignSpec(
        key="fashion_autoresearch",
        task="fashion_mnist",
        comparison="fashion_same_task",
        interface="continuous_autoresearch",
        framework_id="karpathy_autoresearch",
        path=REPO / "data/c0c3/fashion-mnist-autoresearch-v1-7-mps-campaign",
        blocks=(1, 2, 3, 4),
        horizon=44,
    ),
    CampaignSpec(
        key="fashion_bounded_greedy",
        task="fashion_mnist",
        comparison="fashion_same_task",
        interface="bounded_greedy_patch",
        framework_id="openevolve",
        path=REPO / "data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign",
        blocks=(1, 2, 3, 4),
        horizon=44,
    ),
    CampaignSpec(
        key="nanogpt_autoresearch",
        task="nanogpt",
        comparison="nanogpt_early",
        interface="continuous_autoresearch",
        framework_id="karpathy_autoresearch",
        path=REPO / "data/c0c3/nanogpt-autoresearch-v1-7-h100-campaign",
        blocks=(1, 2, 3),
        horizon=5,
    ),
    CampaignSpec(
        key="nanogpt_bounded_greedy",
        task="nanogpt",
        comparison="nanogpt_early",
        interface="bounded_greedy_patch",
        framework_id="openevolve",
        path=REPO / "data/c0c3/nanogpt-openevolve-v2-1-h100-campaign",
        blocks=(1, 2, 3),
        horizon=5,
    ),
    CampaignSpec(
        key="tiny_bounded_greedy",
        task="tiny_adderboard",
        comparison="tiny_open_evolve",
        interface="bounded_greedy_patch",
        framework_id="openevolve",
        path=REPO / "data/c0c3/unified-v3-tiny-adderboard-greedy-campaign",
        blocks=(1, 2, 3, 4, 5, 6, 7, 8),
        horizon=70,
    ),
    CampaignSpec(
        key="tiny_native_population",
        task="tiny_adderboard",
        comparison="tiny_open_evolve",
        interface="native_population",
        framework_id="native_openevolve",
        path=REPO / "data/c0c3/unified-v3-tiny-adderboard-native-campaign",
        blocks=(1, 2, 3, 4, 5, 6, 7, 8),
        horizon=70,
    ),
)

MESSAGE_FIELDS = ("mechanism", "hypothesis", "intended_edit", "evidence")
WORD_RE = re.compile(r"[a-z][a-z0-9_-]+", re.I)
NUMBER_RE = re.compile(r"\d")
LOCAL_PATH_RE = re.compile(r"(/private/|/Users/|LOCAL_TEMP_PATH|REPOSITORY_ROOT)")
UPDATE_PHRASE_RE = re.compile(r"\b(updated|edited|modified)\s+\[?[A-Za-z0-9_.-]*train\.py\b", re.I)
SYNTAX_ONLY_RE = re.compile(r"\bsyntax (?:was )?checked\b|\bnot run\b|\btraining and validation were not run\b", re.I)
ASSUMPTION_RE = re.compile(r"\b(assumption|load-bearing|challenge|alternative|different mechanism|step back)\b", re.I)
PRIOR_RE = re.compile(r"\b(previous|prior|earlier|history|recent result|evidence from)\b", re.I)

FAMILY_PATTERNS: dict[str, re.Pattern[str]] = {
    "conv_spatial": re.compile(r"\b(conv|convolution|kernel|pool|spatial|augmentation|flip|translate)\b", re.I),
    "normalization": re.compile(r"\b(batchnorm|layernorm|norm|affine|bias|scale|temperature|calibration)\b", re.I),
    "optimizer": re.compile(r"\b(optimizer|adam|learning rate|lr|schedule|warmup|batch|gradient|step)\b", re.I),
    "width_capacity": re.compile(r"\b(width|channel|hidden|rank|bottleneck|capacity|dimension|dim)\b", re.I),
    "attention": re.compile(r"\b(attention|head|query|key|value|qkv|routing)\b", re.I),
    "depth": re.compile(r"\b(depth|layer|block|stage|residual|recurrent|loop)\b", re.I),
    "token_interface": re.compile(r"\b(token|embedding|vocab|codebook|symbol|pair)\b", re.I),
}

TEXT_CACHE: dict[Path, str] = {}
NGRAM_CACHE: dict[Path, set[tuple[str, ...]]] = {}
AST_CACHE: dict[Path, Counter[str]] = {}
CHANGED_LINES_CACHE: dict[tuple[Path, Path], int] = {}


@dataclass
class Run:
    spec: CampaignSpec
    run_dir: Path
    manifest: dict[str, Any]
    assignment: dict[str, Any]
    all_events: list[dict[str, Any]]
    events: list[dict[str, Any]]

    @property
    def run_id(self) -> str:
        return str(self.assignment["run_id"])

    @property
    def block(self) -> int:
        return int(self.assignment["block"])

    @property
    def condition(self) -> str:
        return str(self.assignment["condition"])


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL {path}:{number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected object in JSONL {path}:{number}")
        rows.append(value)
    return rows


def load_runs() -> list[Run]:
    output: list[Run] = []
    for spec in CAMPAIGNS:
        if not (spec.path / "campaign.json").is_file():
            raise FileNotFoundError(spec.path / "campaign.json")
        for run_dir in sorted((spec.path / "runs").iterdir()):
            manifest_path = run_dir / "manifest.json"
            events_path = run_dir / "events.jsonl"
            if not (run_dir.is_dir() and manifest_path.is_file() and events_path.is_file()):
                continue
            manifest = load_json(manifest_path)
            assignment = manifest.get("assignment") or {}
            if int(assignment.get("block") or 0) not in spec.blocks:
                continue
            all_events = load_jsonl(events_path)
            events = sorted(
                [row for row in all_events if row.get("event") == "proposal_completed"],
                key=lambda row: int(row["opportunity"]),
            )
            if len(events) < spec.horizon:
                raise ValueError(f"{run_dir.name}: only {len(events)} proposal events, needs horizon {spec.horizon}")
            events = [row for row in events if int(row["opportunity"]) <= spec.horizon]
            opportunities = [int(row["opportunity"]) for row in events]
            if opportunities != list(range(1, spec.horizon + 1)):
                raise ValueError(f"{run_dir.name}: noncontiguous events through {spec.horizon}")
            output.append(Run(spec, run_dir, manifest, assignment, all_events, events))
    expected = {
        "fashion_autoresearch": 16,
        "fashion_bounded_greedy": 16,
        "nanogpt_autoresearch": 12,
        "nanogpt_bounded_greedy": 12,
        "tiny_bounded_greedy": 32,
        "tiny_native_population": 32,
    }
    counts = Counter(run.spec.key for run in output)
    for key, value in expected.items():
        if counts[key] != value:
            raise ValueError(f"{key}: expected {value} runs, found {counts[key]}")
    return output


def usage_total(value: dict[str, Any] | None) -> int:
    value = value or {}
    if value.get("total_tokens") is not None:
        return int(value.get("total_tokens") or 0)
    return int(value.get("input_tokens") or 0) + int(value.get("output_tokens") or 0)


def metric_value(run: Run, record: dict[str, Any], candidate_id: str) -> float:
    evaluation = record.get("evaluation") or {}
    metrics = record.get("metrics") or evaluation.get("metrics") or {}
    if run.spec.task == "fashion_mnist":
        correct = metrics.get("validation_correct")
        return float(correct) if correct is not None else math.nan
    if run.spec.task == "nanogpt":
        bpb = metrics.get("val_bpb")
        return -float(bpb) if bpb is not None else math.nan
    if run.spec.task == "tiny_adderboard":
        parameters = metrics.get("parameters")
        accuracy = float(metrics.get("accuracy") or 0.0)
        valid = bool(evaluation.get("valid", record.get("valid", True if metrics else False)))
        return -float(parameters) if parameters is not None and valid and accuracy >= 0.99 else math.nan
    raise ValueError(run.spec.task)


def baseline_objective(run: Run) -> float:
    baseline = run.manifest["baseline"]
    return metric_value(run, baseline, str(baseline["candidate_id"]))


def candidate_table(run: Run) -> dict[str, dict[str, Any]]:
    baseline = run.manifest["baseline"]
    table: dict[str, dict[str, Any]] = {str(baseline["candidate_id"]): baseline}
    for event in run.all_events:
        candidate_id = str(event.get("candidate_id") or "")
        if not candidate_id:
            continue
        table[candidate_id] = {
            "candidate_id": candidate_id,
            "evaluation": event.get("evaluation") or {},
            "metrics": (event.get("evaluation") or {}).get("metrics") or {},
            "fitness": (event.get("evaluation") or {}).get("fitness"),
        }
    return table


def candidate_objective(run: Run, candidate_id: str, table: dict[str, dict[str, Any]]) -> float:
    value = table.get(candidate_id)
    return metric_value(run, value or {}, candidate_id) if value else math.nan


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
    if path in NGRAM_CACHE:
        return NGRAM_CACHE[path]
    text = TEXT_CACHE.setdefault(path, path.read_text(encoding="utf-8", errors="replace"))
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
        NGRAM_CACHE[path] = set()
        return set()
    result = {tuple(values[index:index + n]) for index in range(max(0, len(values) - n + 1))}
    NGRAM_CACHE[path] = result
    return result


def ast_multiset(path: Path) -> Counter[str]:
    if not path.is_file():
        return Counter()
    if path in AST_CACHE:
        return AST_CACHE[path]
    try:
        text = TEXT_CACHE.setdefault(path, path.read_text(encoding="utf-8", errors="replace"))
        tree = ast.parse(text)
    except SyntaxError:
        AST_CACHE[path] = Counter()
        return Counter()
    result = Counter(type(node).__name__ for node in ast.walk(tree))
    AST_CACHE[path] = result
    return result


def jaccard_distance(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return 1.0 - len(left & right) / len(union) if union else 0.0


def ast_distance(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    denominator = sum(max(left[key], right[key]) for key in keys)
    numerator = sum(abs(left[key] - right[key]) for key in keys)
    return numerator / denominator if denominator else 0.0


def changed_lines(left: Path, right: Path) -> int:
    if not (left.is_file() and right.is_file()):
        return 0
    key = (left, right)
    if key in CHANGED_LINES_CACHE:
        return CHANGED_LINES_CACHE[key]
    old = TEXT_CACHE.setdefault(left, left.read_text(encoding="utf-8", errors="replace")).splitlines()
    new = TEXT_CACHE.setdefault(right, right.read_text(encoding="utf-8", errors="replace")).splitlines()
    count = 0
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            count += max(i2 - i1, j2 - j1)
    CHANGED_LINES_CACHE[key] = count
    return count


def text_fields(event: dict[str, Any]) -> dict[str, str]:
    return {field: str(event.get(field) or "") for field in MESSAGE_FIELDS}


def field_present(value: str) -> bool:
    return bool(value.strip()) and value.strip() != "[not recorded]"


def words(text: str) -> set[str]:
    return {value.lower() for value in WORD_RE.findall(text)}


def lexical_novelty(text: str, prior: list[str]) -> float:
    current = words(text)
    if not current:
        return 0.0
    if not prior:
        return 1.0
    return min(jaccard_distance(current, words(value)) for value in prior)


def mechanism_families(text: str) -> set[str]:
    return {name for name, pattern in FAMILY_PATTERNS.items() if pattern.search(text)}


def mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.mean(finite) if finite else math.nan


def terminal_proposals(run: Run) -> int:
    return sum(1 for row in run.all_events if row.get("event") == "proposal_completed")


def event_rows_for_run(run: Run) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = candidate_table(run)
    base_obj = baseline_objective(run)
    base_metrics = run.manifest["baseline"].get("metrics") or {}
    incumbent_before = str(run.manifest["baseline"]["candidate_id"])
    histories: list[str] = []
    output: list[dict[str, Any]] = []
    for event in run.events:
        opportunity = int(event["opportunity"])
        fields = text_fields(event)
        message_text = " ".join(fields.values())
        candidate_id = str(event.get("candidate_id") or "")
        parent_ids = event.get("selected_parent_ids") or event.get("parent_ids") or []
        parent_id = str(parent_ids[0]) if parent_ids else str(event.get("incumbent_before") or incumbent_before)
        incumbent_after = str(event.get("incumbent_after") or incumbent_before)
        evaluation = event.get("evaluation") or {}
        metrics = evaluation.get("metrics") or {}
        source = proposal_source_path(run, event)
        parent_source = source_path(run, parent_id)
        source_ngrams = normalized_python_ngrams(source)
        parent_ngrams = normalized_python_ngrams(parent_source)
        source_novelty = jaccard_distance(source_ngrams, parent_ngrams) if source_ngrams and parent_ngrams else math.nan
        source_ast = ast_multiset(source)
        parent_ast = ast_multiset(parent_source)
        ast_delta = ast_distance(source_ast, parent_ast) if source_ast and parent_ast else math.nan
        present_fields = sum(field_present(value) for value in fields.values())
        families = mechanism_families(message_text)
        usage = event.get("usage_increment") or {}
        parent_obj = candidate_objective(run, parent_id, table)
        candidate_obj = candidate_objective(run, candidate_id, table)
        incumbent_obj = candidate_objective(run, incumbent_after, table)
        visible = event.get("visible_candidate_ids") or []
        non_incumbent = bool(parent_id and parent_id != str(event.get("incumbent_before") or incumbent_before))
        if run.spec.task == "fashion_mnist":
            endpoint_measure = float(metrics.get("validation_correct") or math.nan)
        elif run.spec.task == "nanogpt":
            endpoint_measure = float(metrics.get("val_bpb") or math.nan)
        else:
            endpoint_measure = float(metrics.get("parameters") or math.nan)
        output.append(
            {
                "comparison": run.spec.comparison,
                "task": run.spec.task,
                "interface": run.spec.interface,
                "framework_id": run.spec.framework_id,
                "campaign": run.spec.path.name,
                "run_id": run.run_id,
                "block": run.block,
                "condition": run.condition,
                "opportunity": opportunity,
                "horizon": run.spec.horizon,
                "candidate_id": candidate_id,
                "parent_id": parent_id,
                "incumbent_after": incumbent_after,
                "valid": int(bool(evaluation.get("valid"))),
                "retained": int(bool(event.get("retained"))),
                "timeout": int(evaluation.get("failure_kind") == "timeout"),
                "failure_kind": str(evaluation.get("failure_kind") or ""),
                "candidate_source_available": int(source.is_file()),
                "parent_source_available": int(parent_source.is_file()),
                "source_novelty": source_novelty,
                "ast_distance": ast_delta,
                "changed_lines": changed_lines(parent_source, source),
                "field_present_count": present_fields,
                "field_complete": int(present_fields == len(MESSAGE_FIELDS)),
                "mechanism_recorded": int(field_present(fields["mechanism"])),
                "evidence_recorded": int(field_present(fields["evidence"])),
                "hypothesis_recorded": int(field_present(fields["hypothesis"])),
                "mechanism_lexical_novelty": lexical_novelty(fields["mechanism"], histories),
                "message_family_count": len(families),
                "mechanism_families": ";".join(sorted(families)),
                "numeric_evidence_field": int(bool(NUMBER_RE.search(fields["evidence"]))),
                "numeric_any_field": int(bool(NUMBER_RE.search(message_text))),
                "prior_language": int(bool(PRIOR_RE.search(message_text))),
                "assumption_language": int(bool(ASSUMPTION_RE.search(message_text))),
                "local_path_marker": int(bool(LOCAL_PATH_RE.search(message_text))),
                "editor_update_phrase": int(bool(UPDATE_PHRASE_RE.search(message_text))),
                "syntax_only_language": int(bool(SYNTAX_ONLY_RE.search(message_text))),
                "tokens": usage_total(usage),
                "input_tokens": int(usage.get("input_tokens") or 0),
                "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
                "reasoning_output_tokens": int(usage.get("reasoning_output_tokens") or 0),
                "evaluator_seconds": float(event.get("evaluator_seconds_increment") or 0.0),
                "visible_candidate_count": len(visible),
                "selected_non_incumbent_parent": int(non_incumbent),
                "conversation_session_id_present": int(bool(event.get("conversation_session_id"))),
                "candidate_objective": candidate_obj,
                "parent_objective": parent_obj,
                "incumbent_objective": incumbent_obj,
                "candidate_measure": endpoint_measure,
            }
        )
        histories.append(fields["mechanism"] or message_text)
        incumbent_before = incumbent_after

    final_event = run.events[-1]
    final_candidate = str(final_event.get("incumbent_after") or incumbent_before)
    final_obj = candidate_objective(run, final_candidate, table)
    endpoint_delta = final_obj - base_obj
    if run.spec.task == "fashion_mnist":
        baseline_value = float(base_metrics["validation_correct"])
        endpoint_value = final_obj
        endpoint_progress = endpoint_delta / max(1.0, 10000.0 - baseline_value)
    elif run.spec.task == "nanogpt":
        baseline_value = float(base_metrics["val_bpb"])
        endpoint_value = -final_obj
        endpoint_progress = endpoint_delta / max(1e-9, baseline_value)
    else:
        baseline_value = float(base_metrics["parameters"])
        endpoint_value = -final_obj
        endpoint_progress = endpoint_delta / max(1.0, baseline_value)

    sessions = {row.get("conversation_session_id") for row in run.events if row.get("conversation_session_id")}
    valid_rows = [row for row in output if row["candidate_source_available"] and row["parent_source_available"]]
    trajectory = {
        "comparison": run.spec.comparison,
        "task": run.spec.task,
        "interface": run.spec.interface,
        "framework_id": run.spec.framework_id,
        "campaign": run.spec.path.name,
        "run_id": run.run_id,
        "block": run.block,
        "condition": run.condition,
        "horizon": run.spec.horizon,
        "terminal_proposals": terminal_proposals(run),
        "baseline_value": baseline_value,
        "endpoint_value": endpoint_value,
        "endpoint_delta": endpoint_delta,
        "endpoint_progress": endpoint_progress,
        "valid_rate": statistics.mean(row["valid"] for row in output),
        "retained_rate": statistics.mean(row["retained"] for row in output),
        "timeout_rate": statistics.mean(row["timeout"] for row in output),
        "source_available_rate": statistics.mean(row["candidate_source_available"] and row["parent_source_available"] for row in output),
        "mean_source_novelty": mean([row["source_novelty"] for row in valid_rows]),
        "mean_ast_distance": mean([row["ast_distance"] for row in valid_rows]),
        "mean_changed_lines": mean([float(row["changed_lines"]) for row in valid_rows]),
        "field_complete_rate": statistics.mean(row["field_complete"] for row in output),
        "mechanism_recorded_rate": statistics.mean(row["mechanism_recorded"] for row in output),
        "evidence_recorded_rate": statistics.mean(row["evidence_recorded"] for row in output),
        "numeric_evidence_field_rate": statistics.mean(row["numeric_evidence_field"] for row in output),
        "numeric_any_field_rate": statistics.mean(row["numeric_any_field"] for row in output),
        "prior_language_rate": statistics.mean(row["prior_language"] for row in output),
        "assumption_language_rate": statistics.mean(row["assumption_language"] for row in output),
        "local_path_marker_rate": statistics.mean(row["local_path_marker"] for row in output),
        "editor_update_phrase_rate": statistics.mean(row["editor_update_phrase"] for row in output),
        "syntax_only_language_rate": statistics.mean(row["syntax_only_language"] for row in output),
        "tokens": sum(row["tokens"] for row in output),
        "tokens_per_proposal": mean([float(row["tokens"]) for row in output]),
        "input_tokens_per_proposal": mean([float(row["input_tokens"]) for row in output]),
        "cached_input_tokens_per_proposal": mean([float(row["cached_input_tokens"]) for row in output]),
        "output_tokens_per_proposal": mean([float(row["output_tokens"]) for row in output]),
        "reasoning_output_tokens_per_proposal": mean([float(row["reasoning_output_tokens"]) for row in output]),
        "evaluator_seconds": sum(row["evaluator_seconds"] for row in output),
        "evaluator_seconds_per_proposal": mean([float(row["evaluator_seconds"]) for row in output]),
        "mean_visible_candidate_count": mean([float(row["visible_candidate_count"]) for row in output]),
        "non_incumbent_parent_rate": statistics.mean(row["selected_non_incumbent_parent"] for row in output),
        "recorded_session_ids": len(sessions),
        "persistent_session_recorded": int(len(sessions) == 1 and all(row["conversation_session_id_present"] for row in output)),
    }
    return output, trajectory


def summary_rows(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        grouped[(str(row["comparison"]), str(row["interface"]))].append(row)
    metrics = (
        "horizon", "terminal_proposals", "endpoint_delta", "endpoint_progress",
        "valid_rate", "retained_rate", "timeout_rate", "mean_source_novelty",
        "mean_changed_lines", "field_complete_rate", "local_path_marker_rate",
        "syntax_only_language_rate", "tokens_per_proposal", "cached_input_tokens_per_proposal",
        "output_tokens_per_proposal", "evaluator_seconds_per_proposal", "mean_visible_candidate_count",
        "non_incumbent_parent_rate", "persistent_session_recorded",
    )
    output: list[dict[str, Any]] = []
    for (comparison, interface), values in sorted(grouped.items()):
        item: dict[str, Any] = {"comparison": comparison, "interface": interface, "runs": len(values)}
        for metric in metrics:
            item[f"{metric}_mean"] = mean([float(row[metric]) for row in values])
        output.append(item)
    return output


def pairwise_contrasts(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["comparison"], row["interface"], int(row["block"]), row["condition"]): row for row in trajectories}
    comparisons = [
        ("fashion_same_task", "bounded_greedy_patch", "continuous_autoresearch", "bounded_minus_continuous", (1, 2, 3, 4), ("C0", "C1", "C2", "C3")),
        ("nanogpt_early", "bounded_greedy_patch", "continuous_autoresearch", "bounded_minus_continuous", (1, 2, 3), ("C0", "C1", "C2", "C3")),
        ("tiny_open_evolve", "native_population", "bounded_greedy_patch", "native_minus_greedy", (1, 2, 3, 4, 5, 6, 7, 8), ("C0", "C1", "C2", "C3")),
    ]
    metrics = (
        "endpoint_delta", "endpoint_progress", "terminal_proposals", "valid_rate", "retained_rate",
        "timeout_rate", "mean_source_novelty", "mean_changed_lines", "field_complete_rate",
        "local_path_marker_rate", "syntax_only_language_rate", "tokens_per_proposal",
        "cached_input_tokens_per_proposal", "output_tokens_per_proposal", "evaluator_seconds_per_proposal",
        "mean_visible_candidate_count", "non_incumbent_parent_rate", "persistent_session_recorded",
    )
    output: list[dict[str, Any]] = []
    for comparison, left_interface, right_interface, direction, blocks, conditions in comparisons:
        for block in blocks:
            for condition in conditions:
                left = lookup[(comparison, left_interface, block, condition)]
                right = lookup[(comparison, right_interface, block, condition)]
                for metric in metrics:
                    lv = float(left[metric])
                    rv = float(right[metric])
                    if math.isfinite(lv) and math.isfinite(rv):
                        output.append(
                            {
                                "comparison": comparison,
                                "direction": direction,
                                "metric": metric,
                                "block": block,
                                "condition": condition,
                                "left_interface": left_interface,
                                "right_interface": right_interface,
                                "left": lv,
                                "right": rv,
                                "difference": lv - rv,
                            }
                        )
    return output


def block_bootstrap(rows: list[dict[str, Any]], *, seed: int, samples: int = 20000) -> tuple[float, float]:
    by_block: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row["difference"])
        if math.isfinite(value):
            by_block[int(row["block"])].append(value)
    blocks = sorted(by_block)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(blocks) for _ in blocks]
        block_means = [sum(by_block[block]) / len(by_block[block]) for block in sampled]
        values.append(sum(block_means) / len(block_means))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def contrast_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["comparison"]), str(row["direction"]), str(row["metric"]))].append(row)
    output: list[dict[str, Any]] = []
    for index, ((comparison, direction, metric), values) in enumerate(sorted(grouped.items())):
        diffs = [float(row["difference"]) for row in values]
        low, high = block_bootstrap(values, seed=SEED + index)
        output.append(
            {
                "comparison": comparison,
                "direction": direction,
                "metric": metric,
                "n_pairs": len(values),
                "difference_mean": mean(diffs),
                "bootstrap_low": low,
                "bootstrap_high": high,
                "left_higher": sum(value > 0 for value in diffs),
                "right_higher": sum(value < 0 for value in diffs),
                "ties": sum(value == 0 for value in diffs),
            }
        )
    return output


def prompt_composition_checks(runs: list[Run]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for interface in ("bounded_greedy_patch", "native_population"):
        subset = [run for run in runs if run.spec.comparison == "tiny_open_evolve" and run.spec.interface == interface]
        prompt_count = 0
        contradiction_count = 0
        reference_count = 0
        multiple_reference_count = 0
        for run in subset:
            for event in run.events:
                prompt = run.run_dir / "opportunities" / f"{int(event['opportunity']):04d}" / "prompt.md"
                if not prompt.is_file():
                    continue
                text = prompt.read_text(encoding="utf-8", errors="replace")
                prompt_count += 1
                has_no_reference_sentence = "No reference design is available" in text
                reference_mentions = len(re.findall(r"REFERENCE DESIGN \d+", text))
                if reference_mentions:
                    reference_count += 1
                if reference_mentions > 1:
                    multiple_reference_count += 1
                if has_no_reference_sentence and reference_mentions:
                    contradiction_count += 1
        results[interface] = {
            "prompt_count": prompt_count,
            "reference_prompt_rate": reference_count / prompt_count if prompt_count else math.nan,
            "multiple_reference_prompt_rate": multiple_reference_count / prompt_count if prompt_count else math.nan,
            "no_reference_sentence_with_reference_rate": contradiction_count / prompt_count if prompt_count else math.nan,
            "no_reference_sentence_with_reference_count": contradiction_count,
        }
    return results


def choose_examples(events: list[dict[str, Any]], trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    candidates = [row for row in events if row["comparison"] == "fashion_same_task" and row["interface"] == "continuous_autoresearch" and row["local_path_marker"]]
    if candidates:
        row = candidates[0]
        examples["autoresearch_local_marker"] = {
            "run_id": row["run_id"],
            "opportunity": row["opportunity"],
            "block": row["block"],
            "condition": row["condition"],
            "description": "Autoresearch record includes an editor-action summary with a machine-local path marker.",
        }
    candidates = [row for row in events if row["comparison"] == "nanogpt_early" and row["interface"] == "continuous_autoresearch" and row["syntax_only_language"]]
    if candidates:
        row = candidates[0]
        examples["nanogpt_syntax_only"] = {
            "run_id": row["run_id"],
            "opportunity": row["opportunity"],
            "block": row["block"],
            "condition": row["condition"],
            "description": "Autoresearch record reports syntax checking and not running training/validation inside the subject message before harness evaluation.",
        }
    fashion_pairs = [row for row in pairwise_contrasts(trajectories) if row["comparison"] == "fashion_same_task" and row["metric"] == "endpoint_delta"]
    if fashion_pairs:
        closest = min(fashion_pairs, key=lambda row: abs(float(row["difference"])))
        examples["closest_fashion_endpoint_pair"] = {
            "block": closest["block"],
            "condition": closest["condition"],
            "bounded_minus_continuous_correct_delta": closest["difference"],
            "description": "A matched block-condition pair with nearly identical endpoint correct-count gain despite large token/interface differences.",
        }
    return examples


def protocol_metadata() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in CAMPAIGNS:
        protocol_path = spec.path / "inputs" / "protocol.json"
        protocol = load_json(protocol_path) if protocol_path.is_file() else {}
        model = protocol.get("model") or {}
        budget = protocol.get("budget") or {}
        rows.append({
            "key": spec.key,
            "task": spec.task,
            "interface": spec.interface,
            "framework_id": spec.framework_id,
            "protocol_version": protocol.get("protocol_version"),
            "study_seed": protocol.get("study_seed"),
            "model_name": model.get("name"),
            "reasoning_effort": model.get("reasoning_effort"),
            "service_tier": model.get("service_tier"),
            "sandbox": model.get("sandbox"),
            "approval_policy": model.get("approval_policy"),
            "conversation_mode": protocol.get("conversation_mode"),
            "proposals_budget": budget.get("proposals"),
            "evaluator_timeout_seconds": budget.get("evaluator_timeout_seconds"),
        })
    return rows


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


def contrast_lookup(rows: list[dict[str, Any]], comparison: str, metric: str) -> dict[str, Any]:
    for row in rows:
        if row["comparison"] == comparison and row["metric"] == metric:
            return row
    raise KeyError((comparison, metric))


def summary_lookup(rows: list[dict[str, Any]], comparison: str, interface: str) -> dict[str, Any]:
    for row in rows:
        if row["comparison"] == comparison and row["interface"] == interface:
            return row
    raise KeyError((comparison, interface))


def box(axis: Any, xy: tuple[float, float], width: float, height: float, text: str, *, face: str, edge: str = "#334155") -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8.1, color="#111827")


def plot_interfaces(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.1, 3.05))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.02, 0.93, "Three research interfaces studied as measurement instruments", fontsize=10.4, weight="bold")
    box(axis, (0.04, 0.58), 0.25, 0.18, "Continuous\nAutoresearch\nlong session", face="#eef2ff", edge="#4f46e5")
    box(axis, (0.375, 0.58), 0.25, 0.18, "Bounded greedy\npatch calls\nsingle incumbent", face="#f0fdf4", edge="#16a34a")
    box(axis, (0.71, 0.58), 0.25, 0.18, "Native population\nOpenEvolve\narchive sampler", face="#fff7ed", edge="#f97316")
    axis.text(0.045, 0.42, "Memory: provider session,\nworkspace, long summaries", fontsize=7.25, va="top")
    axis.text(0.38, 0.42, "Memory: selected source,\nvisible archive, recent outcomes", fontsize=7.25, va="top")
    axis.text(0.715, 0.42, "Memory: islands,\ninspirations, migrations", fontsize=7.25, va="top")
    axis.text(0.045, 0.23, "Trace: coarser metadata,\nmore operational context", fontsize=7.25, va="top", color="#334155")
    axis.text(0.38, 0.23, "Trace: structured fields,\nproposal-level patch contract", fontsize=7.25, va="top", color="#334155")
    axis.text(0.715, 0.23, "Trace: parent sampling,\npopulation-composition checks", fontsize=7.25, va="top", color="#334155")
    axis.text(0.02, 0.07, "Same LLM family and task surface do not imply the same scientific-search instrument.", fontsize=8.5, color="#334155")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output / "figure1_interfaces.pdf", bbox_inches="tight")
    fig.savefig(output / "figure1_interfaces.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fashion_pairs(trajectories: list[dict[str, Any]], output: Path) -> None:
    metrics = [
        ("endpoint_delta", "Validation-correct gain", "higher is better"),
        ("tokens_per_proposal", "Tokens per proposal", "log scale"),
        ("field_complete_rate", "Complete MEHI fields", "record completeness"),
        ("local_path_marker_rate", "Host-path marker rate", "record leakage marker"),
    ]
    lookup = {(row["comparison"], row["interface"], int(row["block"]), row["condition"]): row for row in trajectories}
    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.55))
    for axis, (metric, title, subtitle) in zip(axes, metrics):
        values_left: list[float] = []
        values_right: list[float] = []
        for block in (1, 2, 3, 4):
            for condition in ("C0", "C1", "C2", "C3"):
                continuous = lookup[("fashion_same_task", "continuous_autoresearch", block, condition)]
                bounded = lookup[("fashion_same_task", "bounded_greedy_patch", block, condition)]
                cv = float(continuous[metric])
                bv = float(bounded[metric])
                axis.plot([0, 1], [cv, bv], color="#cbd5e1", lw=0.75)
                values_left.append(cv)
                values_right.append(bv)
        axis.scatter([0] * len(values_left), values_left, color="#4f46e5", s=12, zorder=3)
        axis.scatter([1] * len(values_right), values_right, color="#16a34a", s=12, zorder=3)
        axis.set_xticks([0, 1], ["Continuous", "Bounded"], fontsize=7.0)
        axis.set_title(title, fontsize=8.1)
        axis.text(0.5, 0.98, subtitle, transform=axis.transAxes, ha="center", va="top", fontsize=6.4, color="#64748b")
        axis.grid(axis="y", color="#e5e7eb", lw=0.5)
        if metric == "tokens_per_proposal":
            axis.set_yscale("log")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(output / "figure2_fashion_pairs.pdf", bbox_inches="tight")
    fig.savefig(output / "figure2_fashion_pairs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_extension(trajectory_summary: list[dict[str, Any]], output: Path) -> None:
    panels = [
        ("nanogpt_early", ("continuous_autoresearch", "bounded_greedy_patch"), "NanoGPT first 5 proposals"),
        ("tiny_open_evolve", ("bounded_greedy_patch", "native_population"), "Tiny Addition first 70 proposals"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.6))
    colors = {"continuous_autoresearch": "#4f46e5", "bounded_greedy_patch": "#16a34a", "native_population": "#f97316"}
    for axis, (comparison, interfaces, title) in zip(axes, panels):
        tokens = [summary_lookup(trajectory_summary, comparison, interface)["tokens_per_proposal_mean"] for interface in interfaces]
        endpoint = [summary_lookup(trajectory_summary, comparison, interface)["endpoint_progress_mean"] for interface in interfaces]
        x = np.arange(len(interfaces))
        axis.bar(x - 0.18, tokens, width=0.36, color=[colors[i] for i in interfaces], alpha=0.78, label="tokens/proposal")
        twin = axis.twinx()
        twin.scatter(x + 0.18, endpoint, color="#111827", marker="D", s=28, label="endpoint progress")
        axis.set_yscale("log")
        axis.set_xticks(x, [i.replace("_", "\n") for i in interfaces], fontsize=6.8)
        axis.set_title(title, fontsize=8.3)
        axis.set_ylabel("Tokens/proposal, log", fontsize=7.2)
        twin.set_ylabel("Progress", fontsize=7.2)
        axis.grid(axis="y", color="#e5e7eb", lw=0.5)
    fig.tight_layout()
    fig.savefig(output / "figure3_extensions.pdf", bbox_inches="tight")
    fig.savefig(output / "figure3_extensions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def input_files(runs: list[Run]) -> list[Path]:
    paths: set[Path] = set()
    for run in runs:
        paths.update({
            run.spec.path / "campaign.json",
            run.spec.path / "inputs" / "framework.json",
            run.spec.path / "inputs" / "protocol.json",
            run.spec.path / "inputs" / "task.json",
            run.run_dir / "manifest.json",
            run.run_dir / "events.jsonl",
            run.run_dir / "state.json",
            run.run_dir / "subject-prompt" / "manifest.json",
            run.run_dir / "subject-prompt" / "assumption_changing.md",
        })
        native_root = run.run_dir / "native-openevolve"
        if native_root.is_dir():
            for name in ("events.jsonl", "config.json", "checkpoint.json"):
                path = native_root / name
                if path.is_file():
                    paths.add(path)
        candidate_ids = {str(run.manifest["baseline"].get("candidate_id") or "")}
        for event in run.events:
            opportunity = int(event["opportunity"])
            paths.add(run.run_dir / "opportunities" / f"{opportunity:04d}" / "prompt.md")
            paths.add(run.run_dir / "opportunities" / f"{opportunity:04d}" / "prompt_manifest.json")
            paths.add(run.run_dir / "opportunities" / f"{opportunity:04d}" / "evaluation.json")
            paths.add(run.run_dir / "opportunities" / f"{opportunity:04d}" / "result.json")
            workspace_train = run.run_dir / "opportunities" / f"{opportunity:04d}" / "evaluation-workspace" / "train.py"
            if workspace_train.is_file():
                paths.add(workspace_train)
            for value in [event.get("candidate_id"), event.get("incumbent_before"), event.get("incumbent_after"), *(event.get("selected_parent_ids") or event.get("parent_ids") or [])]:
                if value:
                    candidate_ids.add(str(value))
        for candidate_id in candidate_ids:
            candidate_train = source_path(run, candidate_id)
            if candidate_train.is_file():
                paths.add(candidate_train)
    return sorted(path for path in paths if path.is_file())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(output: Path, files: list[Path], data_root: Path) -> None:
    ledger = output / "input_hashes.json"
    if not ledger.is_file():
        raise ValueError(f"Missing input hash ledger: {ledger}")
    expected = load_json(ledger).get("files") or {}
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
        old = REPO
        REPO = root
        CAMPAIGNS = tuple(
            CampaignSpec(
                key=spec.key,
                task=spec.task,
                comparison=spec.comparison,
                interface=spec.interface,
                framework_id=spec.framework_id,
                path=root / spec.path.relative_to(old),
                blocks=spec.blocks,
                horizon=spec.horizon,
            )
            for spec in CAMPAIGNS
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    files = input_files(runs)
    if args.verify_input_hashes:
        verify_hashes(output, files, REPO)
    all_events: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for run in runs:
        event_rows, trajectory = event_rows_for_run(run)
        all_events.extend(event_rows)
        trajectories.append(trajectory)
    summaries = summary_rows(trajectories)
    pair_rows = pairwise_contrasts(trajectories)
    contrasts = contrast_summary(pair_rows)
    prompt_checks = prompt_composition_checks(runs)
    examples = choose_examples(all_events, trajectories)
    write_csv(output / "event_metrics.csv", all_events)
    write_csv(output / "trajectory_metrics.csv", trajectories)
    write_csv(output / "interface_summary.csv", summaries)
    write_csv(output / "pairwise_contrasts.csv", pair_rows)
    write_csv(output / "contrast_summary.csv", contrasts)
    (output / "prompt_composition_checks.json").write_text(json.dumps(prompt_checks, indent=2, sort_keys=True), encoding="utf-8")
    (output / "qualitative_examples.json").write_text(json.dumps(examples, indent=2, sort_keys=True), encoding="utf-8")
    hashes = {str(path.relative_to(REPO)): sha256(path) for path in files}
    (output / "input_hashes.json").write_text(json.dumps({"algorithm": "sha256", "file_count": len(hashes), "files": hashes}, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "analysis_seed": SEED,
        "campaigns": [
            {
                "key": spec.key,
                "task": spec.task,
                "comparison": spec.comparison,
                "interface": spec.interface,
                "framework_id": spec.framework_id,
                "campaign": spec.path.name,
                "blocks": list(spec.blocks),
                "horizon": spec.horizon,
            }
            for spec in CAMPAIGNS
        ],
        "trajectory_count": len(trajectories),
        "event_count": len(all_events),
        "input_file_count": len(hashes),
        "summary": summaries,
        "contrasts": {f"{row['comparison']}::{row['metric']}": row for row in contrasts},
        "prompt_composition_checks": prompt_checks,
        "protocol_metadata": protocol_metadata(),
        "qualitative_examples": examples,
        "claim_boundary": "Descriptive interface-composed contrasts over recorded traces; proposal rows are not independent experiments.",
    }
    (output / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    plot_interfaces(output)
    plot_fashion_pairs(trajectories, output)
    plot_extension(summaries, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
