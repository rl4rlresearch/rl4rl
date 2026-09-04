#!/usr/bin/env python3
# ruff: noqa: E501
"""Reproduce Paper 6's cross-task assumption-challenge analysis.

The analysis uses three greedy, evaluator-driven code-search campaigns under
protocol 2.1.  The analytic horizons are task specific: 80 proposals for the
10-digit addition transformer, 200 for Fashion-MNIST, and 40 for fixed-time
nanoGPT pretraining.  At every tenth proposal, treated trajectories received
an explicit assumption-challenge direction; matched controls received the
ordinary proposal direction.  Single-incumbent and four-lineage memory are
strata/moderators, not the paper's primary treatment.

Condition labels were fixed rather than randomized and trajectories had
already diverged before the first intervention.  The script therefore reports
matched descriptive contrasts and checkpoint-minus-prior-proposal
difference-in-differences, with block-cluster bootstrap sensitivity intervals.
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import io
import json
import keyword
import math
import random
import re
import statistics
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_OUTPUT = HERE / "derived"
SEED = 20260903

WORD_RE = re.compile(r"[a-z][a-z0-9_+-]{1,}")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", re.I)
ASSUMPTION_RE = re.compile(
    r"\b(?:assum(?:e|ed|es|ing|ption|ptions)|load[- ]bearing|challenge|alternative|"
    r"different (?:mechanism|architecture|representation|computation)|step back|reframe)\b",
    re.I,
)
FAILURE_RE = re.compile(
    r"\b(?:fail(?:ed|ure)?|invalid|could not|unable|collapse[sd]?|regress(?:ed|ion)?|worse|did not|unverified)\b",
    re.I,
)
CAUSAL_RE = re.compile(
    r"\b(?:because|therefore|indicat(?:e|es|ed|ing)|suggest(?:s|ed|ing)?|shows?|implies|motivates?|evidence)\b",
    re.I,
)
MECHANISM_SHIFT_RE = re.compile(
    r"\b(?:instead|rather than|replace|factor|decouple|separate|hierarch|alternative|"
    r"new representation|new mechanism|reframe|challenge|routing|shared|learned pooling)\b",
    re.I,
)

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "will", "into",
    "while", "than", "then", "only", "using", "use", "used", "model",
    "change", "changes", "current", "candidate", "preserve", "increase",
    "reduce", "replace", "add", "adding", "make", "more", "less", "same",
    "final", "existing", "without", "between", "above", "below", "should",
    "would", "could", "because", "through", "over", "under", "after",
    "before", "each", "both", "their", "which", "accuracy", "parameters",
    "training", "steps", "verified", "result", "results", "task", "design",
    "proposal", "opportunity", "learned", "learning", "objective", "score",
}

# Transparent multi-label vocabulary, selected before inspecting aggregate
# effects.  These labels are descriptive aids; source-token and AST distances
# remain the primary novelty measures.
FAMILY_PATTERNS: dict[str, re.Pattern[str]] = {
    "capacity_or_width": re.compile(r"\b(width|dimension|heads?|layers?|feed[- ]?forward|ffn|mlp|capacity|bottleneck|channels?)\b", re.I),
    "attention_or_routing": re.compile(r"\b(attention|qkv|query|key|value|routing|causal|multi[- ]query|head mixing)\b", re.I),
    "token_or_embedding": re.compile(r"\b(token|symbol|embedding|vocab|codebook|lookup|byte|character)\b", re.I),
    "position_or_sequence": re.compile(r"\b(position|positional|relative offset|sequence|context|carry|digit|column|place value)\b", re.I),
    "spatial_representation": re.compile(r"\b(convolution|conv|spatial|pooling|patch|pixel|image|translation|dilation|feature map)\b", re.I),
    "factorization_or_sharing": re.compile(r"\b(tie|tying|factor|factoriz|low[- ]rank|shared|reuse|basis|quotient|gauge)\b", re.I),
    "normalization_or_bias": re.compile(r"\b(layernorm|normalization|batch[- ]?norm|bias|affine|mean[- ]free|offset)\b", re.I),
    "training_procedure": re.compile(r"\b(optimizer|learning rate|schedule|warmup|cosine|batch|gradient|weight decay|initializ|training exposure|step budget)\b", re.I),
    "regularization_or_augmentation": re.compile(r"\b(dropout|regulariz|noise|augment|mixup|cutmix|label smoothing|crop|flip|translation)\b", re.I),
    "ensemble_or_calibration": re.compile(r"\b(ensemble|test[- ]time|multi[- ]view|averag|temperature|calibrat|probability[- ]space|logit blend)\b", re.I),
    "loss_or_objective": re.compile(r"\b(loss|cross[- ]entropy|entropy|margin|auxiliary objective|distill|curriculum)\b", re.I),
}


@dataclass(frozen=True)
class Task:
    key: str
    label: str
    campaign: Path
    horizon: int
    blocks: int
    objective_metric: str
    direction: str
    source_paths: tuple[str, ...]
    objective_label: str


TASKS = (
    Task(
        key="addition",
        label="10-digit addition transformer",
        campaign=REPO / "data/c0c3/controlled-openevolve-transformer-v2-1-mps-campaign",
        horizon=80,
        blocks=5,
        objective_metric="parameters",
        direction="minimize",
        source_paths=("src/model.py", "src/train.py"),
        objective_label="qualified parameter reduction",
    ),
    Task(
        key="fashion",
        label="Fashion-MNIST classifier",
        campaign=REPO / "data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign",
        horizon=200,
        blocks=5,
        objective_metric="validation_score",
        direction="maximize",
        source_paths=("train.py",),
        objective_label="validation-score gain",
    ),
    Task(
        key="nanogpt",
        label="fixed-time language-model pretraining",
        campaign=REPO / "data/c0c3/nanogpt-openevolve-v2-1-h100-campaign",
        horizon=40,
        blocks=3,
        objective_metric="val_bpb",
        direction="minimize",
        source_paths=("train.py",),
        objective_label="validation bits-per-byte reduction",
    ),
)


@dataclass
class Run:
    task: Task
    run_dir: Path
    run_id: str
    block: int
    condition: str
    memory: str
    treated: bool
    baseline: dict[str, Any]
    state: dict[str, Any]
    events: list[dict[str, Any]]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    return rows


def load_runs(task: Task) -> list[Run]:
    runs = []
    for run_dir in sorted((task.campaign / "runs").iterdir()):
        if not (run_dir / "manifest.json").exists():
            continue
        manifest = load_json(run_dir / "manifest.json")
        assignment = manifest["assignment"]
        events = [
            row
            for row in load_jsonl(run_dir / "events.jsonl")
            if row.get("event") == "proposal_completed" and int(row["opportunity"]) <= task.horizon
        ]
        events.sort(key=lambda row: int(row["opportunity"]))
        condition = str(assignment["condition"])
        runs.append(
            Run(
                task=task,
                run_dir=run_dir,
                run_id=str(assignment["run_id"]),
                block=int(assignment["block"]),
                condition=condition,
                memory="four-lineage" if condition in {"C2", "C3"} else "single-incumbent",
                treated=condition in {"C1", "C3"},
                baseline=manifest["baseline"],
                state=load_json(run_dir / "state.json"),
                events=events,
            )
        )
    return runs


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def clean(values: Iterable[Any]) -> list[float]:
    output = []
    for value in values:
        number = safe_float(value)
        if math.isfinite(number):
            output.append(number)
    return output


def mean(values: Iterable[Any]) -> float:
    values = clean(values)
    return statistics.fmean(values) if values else math.nan


def words(text: str) -> set[str]:
    return {word for word in WORD_RE.findall((text or "").lower()) if len(word) > 2 and word not in STOPWORDS}


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def normalized_tokens(source: str) -> list[str]:
    output = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in {tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER, tokenize.COMMENT}:
                continue
            if token.type == tokenize.NAME:
                output.append(token.string if keyword.iskeyword(token.string) else "ID")
            elif token.type == tokenize.NUMBER:
                output.append("NUM")
            elif token.type == tokenize.STRING:
                output.append("STR")
            else:
                output.append(token.string)
    except (IndentationError, tokenize.TokenError):
        return []
    return output


def ngrams(values: Sequence[str], n: int = 3) -> set[tuple[str, ...]]:
    if len(values) < n:
        return {tuple(values)} if values else set()
    return {tuple(values[index : index + n]) for index in range(len(values) - n + 1)}


def ast_nodes(source: str) -> Counter[str]:
    total: Counter[str] = Counter()
    for part in source.split("\n# === FILE BOUNDARY ===\n"):
        try:
            total.update(type(node).__name__ for node in ast.walk(ast.parse(part)))
        except (SyntaxError, ValueError):
            continue
    return total


def counter_distance(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    total = sum(max(left[key], right[key]) for key in keys)
    overlap = sum(min(left[key], right[key]) for key in keys)
    return 0.0 if total == 0 else 1.0 - overlap / total


def diff_count(left: str, right: str) -> int:
    count = 0
    for line in difflib.unified_diff(left.splitlines(), right.splitlines(), n=0):
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")):
            count += 1
    return count


SOURCE_CACHE: dict[tuple[str, str], str | None] = {}


def candidate_source(run: Run, candidate_id: str | None) -> str | None:
    if not candidate_id:
        return None
    key = (run.run_id, candidate_id)
    if key in SOURCE_CACHE:
        return SOURCE_CACHE[key]
    root = run.run_dir / "candidates" / candidate_id
    parts = []
    for relative in run.task.source_paths:
        path = root / relative
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    value = "\n# === FILE BOUNDARY ===\n".join(parts) if len(parts) == len(run.task.source_paths) else None
    SOURCE_CACHE[key] = value
    return value


def event_map(run: Run) -> dict[int, dict[str, Any]]:
    return {int(event["opportunity"]): event for event in run.events}


def candidate_metrics(run: Run, candidate_id: str | None) -> dict[str, Any] | None:
    if not candidate_id:
        return None
    if candidate_id == run.baseline.get("candidate_id"):
        return run.baseline.get("metrics") or {}
    candidate = (run.state.get("candidates") or {}).get(candidate_id)
    if candidate:
        return candidate.get("metrics") or {}
    for event in run.events:
        if event.get("candidate_id") == candidate_id:
            evaluation = event.get("evaluation") or {}
            if evaluation.get("valid"):
                return evaluation.get("metrics") or {}
    return None


def incumbent_metric(run: Run, opportunity: int) -> float:
    event = event_map(run)[opportunity]
    metrics = candidate_metrics(run, event.get("incumbent_after")) or {}
    value = safe_float(metrics.get(run.task.objective_metric))
    if not math.isfinite(value):
        raise ValueError(f"{run.run_id} opportunity {opportunity}: missing incumbent {run.task.objective_metric}")
    return value


def family_tags(text: str) -> set[str]:
    return {name for name, pattern in FAMILY_PATTERNS.items() if pattern.search(text)}


def event_text(event: dict[str, Any]) -> str:
    return " ".join(str(event.get(field) or "") for field in ("mechanism", "hypothesis", "intended_edit", "evidence"))


def message_text(run: Run, opportunity: int) -> str:
    folder = run.run_dir / "opportunities" / f"{opportunity:04d}" / "codex"
    matches = sorted(folder.glob("*.last-message.md"))
    return matches[0].read_text(encoding="utf-8", errors="replace") if matches else ""


def usage_value(event: dict[str, Any], key: str) -> int:
    usage = event.get("usage_increment") or {}
    if key == "total_tokens" and usage.get(key) is None:
        return int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0)
    return int(usage.get(key, 0) or 0)


def validate(runs: Sequence[Run]) -> dict[str, Any]:
    integrity: dict[str, Any] = {
        "tasks": {},
        "total_runs": 0,
        "total_proposals": 0,
        "checkpoint_opportunities": 0,
        "checkpoint_messages_available": 0,
        "checkpoint_provider_failures": 0,
    }
    for task in TASKS:
        selected = [run for run in runs if run.task == task]
        if len(selected) != task.blocks * 4:
            raise ValueError(f"{task.key}: expected {task.blocks * 4} runs, found {len(selected)}")
        assignments = {(run.block, run.condition) for run in selected}
        expected = {(block, condition) for block in range(1, task.blocks + 1) for condition in ("C0", "C1", "C2", "C3")}
        if assignments != expected:
            raise ValueError(f"{task.key}: assignment mismatch")
        for run in selected:
            observed = set(event_map(run))
            missing = set(range(1, task.horizon + 1)) - observed
            if missing:
                raise ValueError(f"{run.run_id}: missing proposals {sorted(missing)}")
            for opportunity in range(10, task.horizon + 1, 10):
                event = event_map(run)[opportunity]
                expected_type = "assumption_changing" if run.treated else "ordinary"
                if event.get("proposal_type") != expected_type:
                    raise ValueError(f"{run.run_id} p{opportunity}: expected {expected_type}")
                prompt = (run.run_dir / "opportunities" / f"{opportunity:04d}" / "prompt.md").read_text(encoding="utf-8")
                has_direction = "step back from the current line of work" in prompt.lower()
                if has_direction != run.treated:
                    raise ValueError(f"{run.run_id} p{opportunity}: treatment prompt mismatch")
                # Provider failures can complete an opportunity without a final
                # agent message. Keep those failures in the executable-outcome
                # analysis and report their absence from the message corpus.
                integrity["checkpoint_opportunities"] += 1
                if message_text(run, opportunity):
                    integrity["checkpoint_messages_available"] += 1
                else:
                    integrity["checkpoint_provider_failures"] += 1
        task_proposals = len(selected) * task.horizon
        integrity["tasks"][task.key] = {
            "runs": len(selected),
            "blocks": task.blocks,
            "horizon": task.horizon,
            "proposals": task_proposals,
            "intervention_checkpoints_per_treated_run": task.horizon // 10,
        }
        integrity["total_runs"] += len(selected)
        integrity["total_proposals"] += task_proposals
    return integrity


def proposal_record(run: Run, event: dict[str, Any]) -> dict[str, Any]:
    opportunity = int(event["opportunity"])
    mapping = event_map(run)
    parent_ids = event.get("selected_parent_ids") or event.get("parent_ids") or []
    parent_id = parent_ids[0] if parent_ids else None
    candidate_id = event.get("candidate_id")
    parent = candidate_source(run, parent_id)
    candidate = candidate_source(run, candidate_id)
    source_available = parent is not None and candidate is not None
    parent_tokens = normalized_tokens(parent or "")
    candidate_tokens = normalized_tokens(candidate or "")
    text = event_text(event)
    prior_events = [mapping[index] for index in range(1, opportunity) if index in mapping]
    current_words = words(text)
    lexical_novelty = min((jaccard(current_words, words(event_text(previous))) for previous in prior_events), default=1.0)
    prior_tags: set[str] = set()
    for previous in prior_events:
        prior_tags.update(family_tags(event_text(previous)))
    tags = family_tags(text)
    evaluation = event.get("evaluation") or {}
    incumbent_before = run.baseline.get("candidate_id") if opportunity == 1 else mapping[opportunity - 1].get("incumbent_after")
    before_metrics = candidate_metrics(run, incumbent_before) or {}
    after_metrics = candidate_metrics(run, event.get("incumbent_after")) or {}
    before_objective = safe_float(before_metrics.get(run.task.objective_metric))
    after_objective = safe_float(after_metrics.get(run.task.objective_metric))
    if math.isfinite(before_objective) and math.isfinite(after_objective):
        incumbent_gain = before_objective - after_objective if run.task.direction == "minimize" else after_objective - before_objective
    else:
        incumbent_gain = math.nan
    evidence = str(event.get("evidence") or "")
    message = message_text(run, opportunity)
    return {
        "task": run.task.key,
        "task_label": run.task.label,
        "run_id": run.run_id,
        "block": run.block,
        "condition": run.condition,
        "memory": run.memory,
        "treated": int(run.treated),
        "opportunity": opportunity,
        "checkpoint": int(opportunity % 10 == 0),
        "intervention": int(run.treated and opportunity % 10 == 0),
        "proposal_type": event.get("proposal_type") or "",
        "candidate_id": candidate_id or "",
        "parent_id": parent_id or "",
        "source_available": int(source_available),
        "source_novelty": jaccard(ngrams(parent_tokens), ngrams(candidate_tokens)) if source_available else math.nan,
        "ast_distance": counter_distance(ast_nodes(parent or ""), ast_nodes(candidate or "")) if source_available else math.nan,
        "changed_lines": diff_count(parent or "", candidate or "") if source_available else math.nan,
        "lexical_novelty": lexical_novelty,
        "family_tags": ";".join(sorted(tags)),
        "new_family_tag": int(bool(tags - prior_tags)),
        "assumption_language": int(bool(ASSUMPTION_RE.search(text))),
        "mechanism_shift_language": int(bool(MECHANISM_SHIFT_RE.search(text))),
        "failure_evidence": int(bool(FAILURE_RE.search(evidence))),
        "numeric_evidence": int(bool(NUMBER_RE.search(evidence))),
        "causal_evidence_language": int(bool(CAUSAL_RE.search(evidence))),
        "message_words": len(WORD_RE.findall(message)),
        "valid": int(bool(evaluation.get("valid"))),
        "retained": int(bool(event.get("retained"))),
        "incumbent_gain": incumbent_gain,
        "incumbent_changed": int(incumbent_gain > 0 if math.isfinite(incumbent_gain) else False),
        "total_tokens": usage_value(event, "total_tokens"),
        "input_tokens": usage_value(event, "input_tokens"),
        "cached_input_tokens": usage_value(event, "cached_input_tokens"),
        "output_tokens": usage_value(event, "output_tokens"),
        "reasoning_output_tokens": usage_value(event, "reasoning_output_tokens"),
        "evaluator_seconds": (
            safe_float(event.get("evaluator_seconds_increment"))
            if math.isfinite(safe_float(event.get("evaluator_seconds_increment")))
            else 0.0
        ),
        "mechanism": event.get("mechanism") or "",
        "hypothesis": event.get("hypothesis") or "",
        "intended_edit": event.get("intended_edit") or "",
        "evidence": evidence,
        "failure_kind": evaluation.get("failure_kind") or "valid",
    }


CHECKPOINT_METRICS = (
    "source_novelty",
    "ast_distance",
    "changed_lines",
    "lexical_novelty",
    "new_family_tag",
    "assumption_language",
    "mechanism_shift_language",
    "failure_evidence",
    "numeric_evidence",
    "causal_evidence_language",
    "message_words",
    "valid",
    "retained",
    "incumbent_changed",
    "incumbent_gain",
    "total_tokens",
    "output_tokens",
    "evaluator_seconds",
)


def checkpoint_pairs(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {(row["task"], int(row["block"]), row["condition"], int(row["opportunity"])): row for row in records}
    output = []
    for task in TASKS:
        for block in range(1, task.blocks + 1):
            for memory, control_condition, treated_condition in (
                ("single-incumbent", "C0", "C1"),
                ("four-lineage", "C2", "C3"),
            ):
                for opportunity in range(10, task.horizon + 1, 10):
                    control = mapping[(task.key, block, control_condition, opportunity)]
                    treated = mapping[(task.key, block, treated_condition, opportunity)]
                    control_before = mapping[(task.key, block, control_condition, opportunity - 1)]
                    treated_before = mapping[(task.key, block, treated_condition, opportunity - 1)]
                    row: dict[str, Any] = {
                        "task": task.key,
                        "block": block,
                        "memory": memory,
                        "opportunity": opportunity,
                        "cycle": opportunity // 10,
                        "control_run_id": control["run_id"],
                        "treated_run_id": treated["run_id"],
                        "control_mechanism": control["mechanism"],
                        "treated_mechanism": treated["mechanism"],
                        "control_hypothesis": control["hypothesis"],
                        "treated_hypothesis": treated["hypothesis"],
                        "control_evidence": control["evidence"],
                        "treated_evidence": treated["evidence"],
                    }
                    for metric in CHECKPOINT_METRICS:
                        cv = safe_float(control[metric])
                        tv = safe_float(treated[metric])
                        cb = safe_float(control_before[metric])
                        tb = safe_float(treated_before[metric])
                        row[f"control_{metric}"] = cv
                        row[f"treated_{metric}"] = tv
                        row[f"difference_{metric}"] = tv - cv if math.isfinite(tv) and math.isfinite(cv) else math.nan
                        row[f"did_{metric}"] = (tv - tb) - (cv - cb) if all(math.isfinite(value) for value in (tv, tb, cv, cb)) else math.nan
                    output.append(row)
    return output


def preintervention_placebos(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the checkpoint DiD to opportunities 2--9, before any treatment."""
    mapping = {(row["task"], int(row["block"]), row["condition"], int(row["opportunity"])): row for row in records}
    output = []
    for task in TASKS:
        for block in range(1, task.blocks + 1):
            for memory, control_condition, treated_condition in (
                ("single-incumbent", "C0", "C1"),
                ("four-lineage", "C2", "C3"),
            ):
                for opportunity in range(2, 10):
                    control = mapping[(task.key, block, control_condition, opportunity)]
                    treated = mapping[(task.key, block, treated_condition, opportunity)]
                    control_before = mapping[(task.key, block, control_condition, opportunity - 1)]
                    treated_before = mapping[(task.key, block, treated_condition, opportunity - 1)]
                    row: dict[str, Any] = {
                        "task": task.key,
                        "block": block,
                        "memory": memory,
                        "opportunity": opportunity,
                    }
                    for metric in CHECKPOINT_METRICS:
                        values = [safe_float(item[metric]) for item in (treated, treated_before, control, control_before)]
                        row[f"did_{metric}"] = (values[0] - values[1]) - (values[2] - values[3]) if all(math.isfinite(value) for value in values) else math.nan
                    output.append(row)
    return output


def placebo_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task.key]
        for metric in CHECKPOINT_METRICS:
            point, low, high = cluster_bootstrap(selected, f"did_{metric}")
            output.append(
                {
                    "task": task.key,
                    "metric": metric,
                    "n_preintervention_pairs": len(selected),
                    "mean_pseudo_did": point,
                    "cluster_bootstrap_low": low,
                    "cluster_bootstrap_high": high,
                }
            )
    return output


def cycle_gain_pairs(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decompose each ten-proposal cycle into checkpoint and follow-up gains."""
    mapping = {(row["task"], int(row["block"]), row["condition"], int(row["opportunity"])): row for row in records}
    output = []
    for task in TASKS:
        for block in range(1, task.blocks + 1):
            for memory, control_condition, treated_condition in (
                ("single-incumbent", "C0", "C1"),
                ("four-lineage", "C2", "C3"),
            ):
                for checkpoint in range(10, task.horizon + 1, 10):
                    end = min(checkpoint + 9, task.horizon)
                    row: dict[str, Any] = {
                        "task": task.key,
                        "block": block,
                        "memory": memory,
                        "checkpoint": checkpoint,
                        "cycle_end": end,
                    }
                    for arm, condition in (("control", control_condition), ("treated", treated_condition)):
                        immediate = safe_float(mapping[(task.key, block, condition, checkpoint)]["incumbent_gain"])
                        followup = sum(
                            safe_float(mapping[(task.key, block, condition, opportunity)]["incumbent_gain"])
                            for opportunity in range(checkpoint + 1, end + 1)
                            if math.isfinite(safe_float(mapping[(task.key, block, condition, opportunity)]["incumbent_gain"]))
                        )
                        row[f"{arm}_immediate_gain"] = immediate
                        row[f"{arm}_followup_gain"] = followup
                        row[f"{arm}_cycle_gain"] = immediate + followup
                    for metric in ("immediate_gain", "followup_gain", "cycle_gain"):
                        row[f"difference_{metric}"] = row[f"treated_{metric}"] - row[f"control_{metric}"]
                    output.append(row)
    return output


def cycle_gain_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task.key]
        for metric in ("immediate_gain", "followup_gain", "cycle_gain"):
            point, low, high = cluster_bootstrap(selected, f"difference_{metric}")
            output.append(
                {
                    "task": task.key,
                    "metric": metric,
                    "n_cycles": len(selected),
                    "control_mean": mean(row[f"control_{metric}"] for row in selected),
                    "treated_mean": mean(row[f"treated_{metric}"] for row in selected),
                    "paired_difference": point,
                    "cluster_bootstrap_low": low,
                    "cluster_bootstrap_high": high,
                }
            )
    return output


def population_dispersion(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Between-run diversity at matched checkpoints, separate from departure."""
    checkpoint = [row for row in records if int(row["checkpoint"])]
    output = []
    for task in TASKS:
        for opportunity in range(10, task.horizon + 1, 10):
            for arm, treated in (("ordinary", 0), ("assumption_challenge", 1)):
                rows = [row for row in checkpoint if row["task"] == task.key and int(row["opportunity"]) == opportunity and int(row["treated"]) == treated]
                lexical = []
                family = []
                for index, left in enumerate(rows):
                    for right in rows[index + 1 :]:
                        lexical.append(jaccard(words(str(left["mechanism"]) + " " + str(left["hypothesis"])), words(str(right["mechanism"]) + " " + str(right["hypothesis"]))))
                        family.append(jaccard(set(str(left["family_tags"]).split(";")) - {""}, set(str(right["family_tags"]).split(";")) - {""}))
                output.append(
                    {
                        "task": task.key,
                        "opportunity": opportunity,
                        "arm": arm,
                        "n_runs": len(rows),
                        "n_pairs": len(lexical),
                        "between_run_lexical_distance": mean(lexical),
                        "between_run_family_distance": mean(family),
                    }
                )
    return output


def cluster_bootstrap(rows: Sequence[dict[str, Any]], field: str, repetitions: int = 10000) -> tuple[float, float, float]:
    by_block: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = safe_float(row.get(field))
        if math.isfinite(value):
            by_block[int(row["block"])].append(value)
    block_means = {block: mean(values) for block, values in by_block.items()}
    keys = sorted(block_means)
    point = mean(block_means.values())
    rng = random.Random(SEED + sum(ord(char) for char in field) + len(rows))
    estimates = []
    for _ in range(repetitions):
        sampled = [block_means[rng.choice(keys)] for _ in keys]
        estimates.append(mean(sampled))
    estimates.sort()
    return point, estimates[int(0.025 * repetitions)], estimates[int(0.975 * repetitions)]


def checkpoint_effects(pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for task in TASKS:
        task_rows = [row for row in pairs if row["task"] == task.key]
        for memory in ("all", "single-incumbent", "four-lineage"):
            selected = task_rows if memory == "all" else [row for row in task_rows if row["memory"] == memory]
            for metric in CHECKPOINT_METRICS:
                point, low, high = cluster_bootstrap(selected, f"did_{metric}")
                output.append(
                    {
                        "task": task.key,
                        "memory": memory,
                        "metric": metric,
                        "n_checkpoint_pairs": len(selected),
                        "control_checkpoint_mean": mean(row[f"control_{metric}"] for row in selected),
                        "treated_checkpoint_mean": mean(row[f"treated_{metric}"] for row in selected),
                        "raw_paired_difference": mean(row[f"difference_{metric}"] for row in selected),
                        "did_effect": point,
                        "cluster_bootstrap_low": low,
                        "cluster_bootstrap_high": high,
                    }
                )
    return output


def memory_moderation(pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for task in TASKS:
        rows = [row for row in pairs if row["task"] == task.key]
        for metric in CHECKPOINT_METRICS:
            by_block: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for row in rows:
                value = safe_float(row[f"did_{metric}"])
                if math.isfinite(value):
                    by_block[int(row["block"])][str(row["memory"])].append(value)
            contrasts = []
            for block, memories in by_block.items():
                if memories["single-incumbent"] and memories["four-lineage"]:
                    contrasts.append({"block": block, "difference": mean(memories["four-lineage"]) - mean(memories["single-incumbent"])})
            pseudo = [{"block": row["block"], "value": row["difference"]} for row in contrasts]
            point, low, high = cluster_bootstrap(pseudo, "value")
            output.append(
                {
                    "task": task.key,
                    "metric": metric,
                    "n_blocks": len(contrasts),
                    "portfolio_minus_single_moderation": point,
                    "cluster_bootstrap_low": low,
                    "cluster_bootstrap_high": high,
                }
            )
    return output


def run_outcome(run: Run, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in records if row["run_id"] == run.run_id]
    start = incumbent_metric(run, 9)
    final = incumbent_metric(run, run.task.horizon)
    progress = start - final if run.task.direction == "minimize" else final - start
    scale = abs(start) if abs(start) > 0 else 1.0
    sign = -1.0 if run.task.direction == "minimize" else 1.0
    return {
        "task": run.task.key,
        "run_id": run.run_id,
        "block": run.block,
        "condition": run.condition,
        "memory": run.memory,
        "treated": int(run.treated),
        "horizon": run.task.horizon,
        "objective_at_9": start,
        "objective_at_horizon": final,
        "fitness_at_9": sign * start,
        "fitness_at_horizon": sign * final,
        "objective_progress": progress,
        "normalized_progress": progress / scale,
        "valid_rate_10_onward": mean(row["valid"] for row in selected if int(row["opportunity"]) >= 10),
        "retained_rate_10_onward": mean(row["retained"] for row in selected if int(row["opportunity"]) >= 10),
        "source_novelty_10_onward": mean(row["source_novelty"] for row in selected if int(row["opportunity"]) >= 10),
        "tokens_10_onward": sum(int(row["total_tokens"]) for row in selected if int(row["opportunity"]) >= 10),
        "evaluator_seconds_10_onward": sum(float(row["evaluator_seconds"]) for row in selected if int(row["opportunity"]) >= 10),
    }


OUTCOME_METRICS = (
    "fitness_at_9",
    "fitness_at_horizon",
    "objective_progress",
    "normalized_progress",
    "valid_rate_10_onward",
    "retained_rate_10_onward",
    "source_novelty_10_onward",
    "tokens_10_onward",
    "evaluator_seconds_10_onward",
)


def endpoint_pairs(outcomes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {(row["task"], int(row["block"]), row["condition"]): row for row in outcomes}
    output = []
    for task in TASKS:
        for block in range(1, task.blocks + 1):
            for memory, control_condition, treated_condition in (
                ("single-incumbent", "C0", "C1"),
                ("four-lineage", "C2", "C3"),
            ):
                control = mapping[(task.key, block, control_condition)]
                treated = mapping[(task.key, block, treated_condition)]
                row: dict[str, Any] = {"task": task.key, "block": block, "memory": memory}
                for metric in OUTCOME_METRICS:
                    row[f"control_{metric}"] = control[metric]
                    row[f"treated_{metric}"] = treated[metric]
                    row[f"difference_{metric}"] = treated[metric] - control[metric]
                output.append(row)
    return output


def endpoint_effects(pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for task in TASKS:
        task_rows = [row for row in pairs if row["task"] == task.key]
        for memory in ("all", "single-incumbent", "four-lineage"):
            selected = task_rows if memory == "all" else [row for row in task_rows if row["memory"] == memory]
            for metric in OUTCOME_METRICS:
                point, low, high = cluster_bootstrap(selected, f"difference_{metric}")
                output.append(
                    {
                        "task": task.key,
                        "memory": memory,
                        "metric": metric,
                        "n_pairs": len(selected),
                        "control_mean": mean(row[f"control_{metric}"] for row in selected),
                        "treated_mean": mean(row[f"treated_{metric}"] for row in selected),
                        "paired_difference": point,
                        "cluster_bootstrap_low": low,
                        "cluster_bootstrap_high": high,
                        "treated_higher": sum(row[f"difference_{metric}"] > 0 for row in selected),
                        "control_higher": sum(row[f"difference_{metric}"] < 0 for row in selected),
                    }
                )
    return output


def message_theme_summary(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    checkpoint = [row for row in records if int(row["checkpoint"])]
    output = []
    for task in TASKS:
        for arm, treated in (("ordinary", 0), ("assumption_challenge", 1)):
            rows = [row for row in checkpoint if row["task"] == task.key and int(row["treated"]) == treated]
            all_families = sorted(FAMILY_PATTERNS)
            for family in all_families:
                output.append(
                    {
                        "task": task.key,
                        "arm": arm,
                        "theme": family,
                        "n": len(rows),
                        "share": mean(family in str(row["family_tags"]).split(";") for row in rows),
                        "valid_rate": mean(row["valid"] for row in rows if family in str(row["family_tags"]).split(";")),
                        "retained_rate": mean(row["retained"] for row in rows if family in str(row["family_tags"]).split(";")),
                    }
                )
    return output


def cycle_profile(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    metrics = ("source_novelty", "ast_distance", "changed_lines", "valid", "retained", "output_tokens", "message_words")
    for task in TASKS:
        rows = [row for row in records if row["task"] == task.key and int(row["opportunity"]) >= 10]
        for offset in range(10):
            # opportunity 10 is offset 0, 11 is 1, ..., 19 is 9.
            selected = [row for row in rows if int(row["opportunity"]) % 10 == offset]
            for treated in (0, 1):
                arm_rows = [row for row in selected if int(row["treated"]) == treated]
                record: dict[str, Any] = {"task": task.key, "offset": offset, "arm": "challenge" if treated else "ordinary", "n": len(arm_rows)}
                for metric in metrics:
                    record[metric] = mean(row[metric] for row in arm_rows)
                output.append(record)
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def effect_lookup(effects: Sequence[dict[str, Any]], task: str, metric: str, memory: str = "all") -> dict[str, Any]:
    return next(row for row in effects if row["task"] == task and row["metric"] == metric and row["memory"] == memory)


def make_figures(checkpoint: Sequence[dict[str, Any]], endpoint: Sequence[dict[str, Any]], cycle: Sequence[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titleweight": "bold"})
    task_colors = {"addition": "#2C5F8A", "fashion": "#B15C2E", "nanogpt": "#4A7C59"}
    labels = {task.key: task.label for task in TASKS}

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15))
    metrics = [
        ("source_novelty", "Source-structure novelty"),
        ("valid", "Executable / qualified"),
        ("output_tokens", "Output tokens"),
    ]
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for index, task in enumerate(TASKS):
            row = effect_lookup(checkpoint, task.key, metric)
            value = float(row["did_effect"])
            low = float(row["cluster_bootstrap_low"])
            high = float(row["cluster_bootstrap_high"])
            axis.errorbar(value, index, xerr=[[value - low], [high - value]], fmt="o", color=task_colors[task.key], capsize=3, markersize=5)
        axis.axvline(0, color="#777777", linewidth=0.8)
        axis.set_yticks(range(len(TASKS)), [labels[task.key] for task in TASKS] if axis is axes[0] else [""] * len(TASKS))
        axis.set_title(title, fontsize=9)
        axis.grid(axis="x", alpha=0.18)
        axis.set_xlabel("challenge effect (local DiD)")
    fig.suptitle("Immediate checkpoint effects at proposals 10, 20, ...", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output / "figure1_checkpoint_effects.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True)
    for axis, task in zip(axes, TASKS, strict=True):
        rows = [row for row in cycle if row["task"] == task.key]
        for arm, style, color in (("ordinary", "--", "#6B7280"), ("challenge", "-", task_colors[task.key])):
            selected = sorted((row for row in rows if row["arm"] == arm), key=lambda row: int(row["offset"]))
            axis.plot([int(row["offset"]) for row in selected], [float(row["source_novelty"]) for row in selected], style, color=color, marker="o", markersize=3, label=arm.replace("_", " "))
        axis.axvline(0, color="#111827", alpha=0.22, linewidth=7)
        axis.set_title(task.label, fontsize=9)
        axis.set_xlabel("proposals since scheduled checkpoint")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("mean source novelty")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Source novelty across each ten-proposal intervention cycle", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output / "figure2_cycle_profile.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.25))
    endpoint_pairs_rows = []
    for task in TASKS:
        task_rows = [row for row in endpoint if row["task"] == task.key and row["metric"] == "normalized_progress" and row["memory"] != "all"]
        endpoint_pairs_rows.extend(task_rows)
    for axis, task in zip(axes, TASKS, strict=True):
        for index, memory in enumerate(("single-incumbent", "four-lineage")):
            row = effect_lookup(endpoint, task.key, "normalized_progress", memory)
            value = 100 * float(row["paired_difference"])
            low = 100 * float(row["cluster_bootstrap_low"])
            high = 100 * float(row["cluster_bootstrap_high"])
            axis.errorbar(value, index, xerr=[[value - low], [high - value]], fmt="o", capsize=3, color=task_colors[task.key], markersize=6)
        axis.axvline(0, color="#777777", linewidth=0.8)
        axis.set_yticks([0, 1], ["single incumbent", "four-lineage memory"] if axis is axes[0] else ["", ""])
        axis.set_title(task.label, fontsize=9)
        axis.set_xlabel("treated - control\nprogress (% of proposal-9 objective)")
        axis.grid(axis="x", alpha=0.18)
    fig.suptitle("Downstream task progress is heterogeneous", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output / "figure3_endpoint_progress.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_qualitative_corpus(records: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]], output: Path) -> None:
    lines = [
        "# Checkpoint message audit corpus",
        "",
        "This file contains every agent-authored mechanism, hypothesis, intended edit, and evidence field at every scheduled intervention checkpoint and its matched ordinary-control checkpoint. It is generated from the saved final messages/result records and is intended for qualitative audit, not as an independent annotation.",
        "",
    ]
    checkpoint = [row for row in records if int(row["checkpoint"])]
    checkpoint.sort(key=lambda row: (row["task"], int(row["block"]), row["memory"], int(row["opportunity"]), int(row["treated"])))
    for row in checkpoint:
        arm = "ASSUMPTION CHALLENGE" if int(row["treated"]) else "ORDINARY CONTROL"
        lines.extend(
            [
                f"## {row['task']} | block {row['block']} | {row['memory']} | proposal {row['opportunity']} | {arm}",
                "",
                f"- Mechanism: {row['mechanism']}",
                f"- Hypothesis: {row['hypothesis']}",
                f"- Intended edit: {row['intended_edit']}",
                f"- Evidence: {row['evidence']}",
                f"- Outcome: valid={row['valid']}; retained={row['retained']}; incumbent_gain={row['incumbent_gain']}",
                "",
            ]
        )
    (output / "checkpoint_message_corpus.md").write_text("\n".join(lines), encoding="utf-8")

    # Deterministic high-information sample: largest source departure, largest
    # accepted gain, and a failed challenge for each task/memory stratum.
    sample_lines = ["# Deterministic qualitative sample", ""]
    for task in TASKS:
        for memory in ("single-incumbent", "four-lineage"):
            rows = [row for row in checkpoint if row["task"] == task.key and row["memory"] == memory and int(row["treated"])]
            selections: list[tuple[str, dict[str, Any]]] = []
            available = [row for row in rows if math.isfinite(safe_float(row["source_novelty"]))]
            if available:
                selections.append(("largest source departure", max(available, key=lambda row: safe_float(row["source_novelty"]))))
            improved = [row for row in rows if safe_float(row["incumbent_gain"]) > 0]
            if improved:
                selections.append(("largest retained improvement", max(improved, key=lambda row: safe_float(row["incumbent_gain"]))))
            failed = [row for row in rows if not int(row["valid"])]
            if failed:
                selections.append(("first invalid alternative", min(failed, key=lambda row: int(row["opportunity"]))))
            seen = set()
            for reason, row in selections:
                key = (row["run_id"], row["opportunity"])
                if key in seen:
                    continue
                seen.add(key)
                sample_lines.extend(
                    [
                        f"## {task.label} | {memory} | {reason}",
                        "",
                        f"Run: `{row['run_id']}`; proposal {row['opportunity']}.",
                        "",
                        f"**Mechanism.** {row['mechanism']}",
                        "",
                        f"**Hypothesis.** {row['hypothesis']}",
                        "",
                        f"**Evidence.** {row['evidence']}",
                        "",
                        f"**Outcome.** valid={row['valid']}; retained={row['retained']}; source_novelty={row['source_novelty']}; incumbent_gain={row['incumbent_gain']}.",
                        "",
                    ]
                )
    (output / "qualitative_sample.md").write_text("\n".join(sample_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    runs = [run for task in TASKS for run in load_runs(task)]
    integrity = validate(runs)
    records = [proposal_record(run, event) for run in runs for event in run.events]
    pairs = checkpoint_pairs(records)
    placebos = preintervention_placebos(records)
    placebo_effects = placebo_summary(placebos)
    cycles = cycle_gain_pairs(records)
    cycle_effects = cycle_gain_summary(cycles)
    dispersion = population_dispersion(records)
    checkpoint = checkpoint_effects(pairs)
    moderation = memory_moderation(pairs)
    outcomes = [run_outcome(run, records) for run in runs]
    end_pairs = endpoint_pairs(outcomes)
    endpoint = endpoint_effects(end_pairs)
    themes = message_theme_summary(records)
    cycle = cycle_profile(records)

    write_csv(output / "proposal_records.csv", records)
    write_csv(output / "checkpoint_pairs.csv", pairs)
    write_csv(output / "checkpoint_effects.csv", checkpoint)
    write_csv(output / "preintervention_placebos.csv", placebos)
    write_csv(output / "preintervention_placebo_effects.csv", placebo_effects)
    write_csv(output / "cycle_gain_pairs.csv", cycles)
    write_csv(output / "cycle_gain_effects.csv", cycle_effects)
    write_csv(output / "population_dispersion.csv", dispersion)
    write_csv(output / "memory_moderation.csv", moderation)
    write_csv(output / "trajectory_outcomes.csv", outcomes)
    write_csv(output / "endpoint_pairs.csv", end_pairs)
    write_csv(output / "endpoint_effects.csv", endpoint)
    write_csv(output / "message_theme_summary.csv", themes)
    write_csv(output / "cycle_profile.csv", cycle)
    write_qualitative_corpus(records, pairs, output)
    make_figures(checkpoint, endpoint, cycle, output)

    overview = {
        "integrity": integrity,
        "analysis_horizons": {task.key: task.horizon for task in TASKS},
        "checkpoint_pairs": len(pairs),
        "intervention_messages": sum(int(row["intervention"]) for row in records),
        "matched_checkpoint_messages": sum(int(row["checkpoint"]) for row in records),
        "source_available_rate": mean(row["source_available"] for row in records),
        "primary_effects": {
            task.key: {
                metric: effect_lookup(checkpoint, task.key, metric)
                for metric in ("source_novelty", "ast_distance", "changed_lines", "valid", "retained", "incumbent_changed", "output_tokens", "evaluator_seconds")
            }
            for task in TASKS
        },
        "preintervention_placebo_effects": {
            task.key: {
                metric: next(row for row in placebo_effects if row["task"] == task.key and row["metric"] == metric)
                for metric in ("source_novelty", "ast_distance", "changed_lines", "valid", "retained", "output_tokens")
            }
            for task in TASKS
        },
        "cycle_gain_effects": {
            task.key: {
                metric: next(row for row in cycle_effects if row["task"] == task.key and row["metric"] == metric)
                for metric in ("immediate_gain", "followup_gain", "cycle_gain")
            }
            for task in TASKS
        },
        "endpoint_effects": {
            task.key: effect_lookup(endpoint, task.key, "normalized_progress") for task in TASKS
        },
    }
    (output / "overview.json").write_text(json.dumps(overview, indent=2, sort_keys=True), encoding="utf-8")
    (output / "integrity.json").write_text(json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(overview, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
