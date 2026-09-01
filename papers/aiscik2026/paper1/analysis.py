#!/usr/bin/env python3
# ruff: noqa: E501
"""Reproducible Paper 1 audit of the completed Fashion-MNIST campaign.

The script uses only Python's standard library plus NumPy and Matplotlib. It
parses every proposal-completed event, every available agent final message,
and every available candidate/parent source snapshot. All derived tables and
figures are written beneath ``derived/`` next to this file.
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import io
import json
import keyword
import math
import os
import random
import re
import statistics
import tokenize
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_CAMPAIGN = (
    REPO / "data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign"
)
DEFAULT_OUTPUT = HERE / "derived"
SEED = 20260901

WORD_RE = re.compile(r"[a-z][a-z0-9_+-]{1,}")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", re.I)
PRESERVE_PREDICTIONS_RE = re.compile(
    r"(?:preserv(?:e|ing|es|ed).{0,45}(?:correct|prediction|argmax)|"
    r"same.{0,35}(?:correct|prediction|argmax)|"
    r"without chang(?:e|ing).{0,35}(?:correct|prediction|argmax)|"
    r"keep(?:ing|s)?.{0,35}(?:correct|prediction|argmax))",
    re.I,
)
LOWER_CE_RE = re.compile(
    r"(?:lower|reduce|improv|minimi[sz]).{0,50}(?:cross.?entropy|\bce\b|loss)",
    re.I,
)
EVIDENCE_UPDATE_RE = re.compile(
    r"(?:previous|prior|recent|last|achiev|improv|failed|worse|better|"
    r"timeout|retained|rejected|verified|result|score|correct|accuracy|"
    r"cross.?entropy|loss|trial|evidence)",
    re.I,
)
CALIBRATION_RE = re.compile(
    r"(?:temperature|calibrat|logit.?scale|ensemble.?weight|mixture.?weight|"
    r"blend.?weight|binary.?search|probability.?mixture)",
    re.I,
)

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "will", "into",
    "while", "than", "then", "only", "using", "use", "used", "model",
    "change", "changes", "validation", "training", "current", "candidate",
    "preserve", "increase", "reduce", "replace", "add", "adding", "make",
    "more", "less", "same", "final", "existing", "without", "between",
    "above", "below", "should", "would", "could", "because", "through",
    "over", "under", "after", "before", "each", "both", "their", "which",
}


@dataclass
class RunData:
    run_id: str
    block: int
    condition: str
    run_dir: Path
    manifest: dict[str, Any]
    state: dict[str, Any]
    events: list[dict[str, Any]]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{lineno}") from exc
    return rows


def tokenize_words(text: str) -> set[str]:
    return {
        token
        for token in WORD_RE.findall((text or "").lower())
        if token not in STOPWORDS and len(token) > 2
    }


def jaccard_distance(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def declared_novelty(text: str, prior: Sequence[set[str]]) -> float:
    words = tokenize_words(text)
    if not prior:
        return 1.0
    return min(jaccard_distance(words, old) for old in prior)


def normalized_python_tokens(source: str) -> list[str]:
    """Tokenize code while abstracting identifiers, literals, and comments."""
    out: list[str] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in stream:
            kind = tok.type
            value = tok.string
            if kind in {tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE,
                        tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER,
                        tokenize.COMMENT}:
                continue
            if kind == tokenize.NAME:
                out.append(value if keyword.iskeyword(value) else "ID")
            elif kind == tokenize.NUMBER:
                out.append("NUM")
            elif kind == tokenize.STRING:
                out.append("STR")
            else:
                out.append(value)
    except (IndentationError, tokenize.TokenError):
        return []
    return out


def token_ngrams(tokens: Sequence[str], n: int = 3) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def ast_signature(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "syntax_error"
    nodes = Counter(type(node).__name__ for node in ast.walk(tree))
    payload = json.dumps(sorted(nodes.items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def read_source(run_dir: Path, candidate_id: str | None) -> str | None:
    if not candidate_id:
        return None
    path = run_dir / "candidates" / candidate_id / "train.py"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def diff_size(parent: str | None, candidate: str | None) -> tuple[int, int, int]:
    if parent is None or candidate is None:
        return 0, 0, 0
    added = deleted = 0
    for line in difflib.ndiff(parent.splitlines(), candidate.splitlines()):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            deleted += 1
    return added, deleted, added + deleted


def agent_message(run: RunData, opportunity: int) -> str:
    folder = run.run_dir / "opportunities" / f"{opportunity:04d}" / "codex"
    path = folder / f"proposal-{opportunity}.last-message.md"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    jsonl = folder / f"proposal-{opportunity}.jsonl"
    if jsonl.exists():
        for item in reversed(load_jsonl(jsonl)):
            payload = item.get("item") or {}
            if item.get("type") == "item.completed" and payload.get("type") == "agent_message":
                return str(payload.get("text") or "")
    return ""


def discover_runs(campaign: Path) -> list[RunData]:
    runs: list[RunData] = []
    for run_dir in sorted((campaign / "runs").iterdir()):
        if not run_dir.is_dir() or not (run_dir / "manifest.json").exists():
            continue
        manifest = load_json(run_dir / "manifest.json")
        assignment = manifest["assignment"]
        events = [
            row for row in load_jsonl(run_dir / "events.jsonl")
            if row.get("event") == "proposal_completed"
        ]
        events.sort(key=lambda row: int(row["opportunity"]))
        runs.append(RunData(
            run_id=assignment["run_id"],
            block=int(assignment["block"]),
            condition=assignment["condition"],
            run_dir=run_dir,
            manifest=manifest,
            state=load_json(run_dir / "state.json"),
            events=events,
        ))
    return sorted(runs, key=lambda run: (run.block, run.condition))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def usage_total(usage: dict[str, Any] | None) -> int:
    usage = usage or {}
    if usage.get("total_tokens") is not None:
        return int(usage["total_tokens"])
    return int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0)


def rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    start = 0
    while start < len(arr):
        end = start + 1
        while end < len(arr) and arr[order[end]] == arr[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x = rankdata(left)
    y = rankdata(right)
    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def block_bootstrap_ci(
    rows: Sequence[dict[str, Any]], x_key: str, y_key: str, draws: int = 10000
) -> tuple[float, float]:
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_block[int(row["block"])].append(row)
    blocks = sorted(by_block)
    rng = random.Random(SEED)
    stats: list[float] = []
    for _ in range(draws):
        sampled: list[dict[str, Any]] = []
        for block in (rng.choice(blocks) for _ in blocks):
            sampled.extend(by_block[block])
        value = spearman(
            [safe_float(row[x_key]) for row in sampled],
            [safe_float(row[y_key]) for row in sampled],
        )
        if math.isfinite(value):
            stats.append(value)
    if not stats:
        return math.nan, math.nan
    return tuple(float(x) for x in np.quantile(stats, [0.025, 0.975]))


def block_bootstrap_difference_ci(
    rows: Sequence[dict[str, Any]], pre_key: str, post_key: str, draws: int = 10000
) -> tuple[float, float]:
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_block[int(row["block"])].append(row)
    blocks = sorted(by_block)
    rng = random.Random(SEED + 17)
    stats: list[float] = []
    for _ in range(draws):
        sampled: list[dict[str, Any]] = []
        for block in (rng.choice(blocks) for _ in blocks):
            sampled.extend(by_block[block])
        differences = [
            safe_float(row[post_key]) - safe_float(row[pre_key]) for row in sampled
            if math.isfinite(safe_float(row[post_key])) and math.isfinite(safe_float(row[pre_key]))
        ]
        if differences:
            stats.append(statistics.fmean(differences))
    return tuple(float(x) for x in np.quantile(stats, [0.025, 0.975]))


def summarize_condition(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    measures = [
        "final_correct", "correct_gain", "final_score", "valid_rate",
        "new_best_count", "accuracy_improvement_count", "tiebreak_improvement_count",
        "post_accuracy_plateau_proposals", "post_accuracy_plateau_token_fraction",
        "constant_only_rate", "mean_declared_novelty", "mean_structural_novelty",
        "numeric_evidence_rate", "evidence_update_rate", "preserve_ce_rate",
    ]
    output: list[dict[str, Any]] = []
    for condition in ["C0", "C1", "C2", "C3"]:
        group = [row for row in rows if row["condition"] == condition]
        record: dict[str, Any] = {"condition": condition, "n_runs": len(group)}
        for key in measures:
            values = [safe_float(row[key]) for row in group]
            values = [value for value in values if math.isfinite(value)]
            record[f"{key}_mean"] = statistics.fmean(values) if values else math.nan
            record[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else math.nan
        output.append(record)
    return output


def analyse(campaign: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(campaign)
    if len(runs) != 20:
        raise ValueError(f"Expected 20 completed runs, found {len(runs)}")

    proposal_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    qualitative_rows: list[dict[str, Any]] = []

    for run in runs:
        if len(run.events) != 200:
            raise ValueError(f"{run.run_id}: expected 200 completed proposals, got {len(run.events)}")
        baseline = run.manifest["baseline"]["metrics"]
        candidate_records = run.state.get("candidates") or {}
        final_candidate = candidate_records[run.state["incumbent_id"]]
        baseline_score = float(baseline["validation_score"])
        baseline_correct = int(baseline["validation_correct"])
        baseline_ce = float(baseline["validation_cross_entropy"])
        best_score = baseline_score
        best_correct = baseline_correct
        best_ce = baseline_ce
        last_score_improvement = 0
        last_correct_improvement = 0
        new_best_count = 0
        accuracy_improvement_count = 0
        tiebreak_improvement_count = 0
        declared_history: list[set[str]] = []
        total_tokens = 0
        total_evaluator_seconds = 0.0

        for event in run.events:
            opportunity = int(event["opportunity"])
            evaluation = event.get("evaluation") or {}
            metrics = evaluation.get("metrics") or {}
            valid = bool(evaluation.get("valid"))
            score = safe_float(metrics.get("validation_score"))
            correct = safe_float(metrics.get("validation_correct"))
            ce = safe_float(metrics.get("validation_cross_entropy"))
            message = agent_message(run, opportunity)
            mechanism = str(event.get("mechanism") or "")
            hypothesis = str(event.get("hypothesis") or "")
            intended_edit = str(event.get("intended_edit") or "")
            evidence = str(event.get("evidence") or "")
            declared_text = " ".join([mechanism, hypothesis, intended_edit])
            novelty = declared_novelty(declared_text, declared_history)
            declared_history.append(tokenize_words(declared_text))

            selected_parent_ids = event.get("selected_parent_ids") or event.get("parent_ids") or []
            parent_id = selected_parent_ids[0] if selected_parent_ids else None
            parent_source = read_source(run.run_dir, parent_id)
            candidate_source = read_source(run.run_dir, event.get("candidate_id"))
            parent_tokens = normalized_python_tokens(parent_source or "")
            candidate_tokens = normalized_python_tokens(candidate_source or "")
            source_available = parent_source is not None and candidate_source is not None
            constant_only = bool(source_available and parent_tokens == candidate_tokens and parent_source != candidate_source)
            no_source_change = bool(source_available and parent_source == candidate_source)
            structural_novelty = (
                jaccard_distance(token_ngrams(parent_tokens), token_ngrams(candidate_tokens))
                if source_available else math.nan
            )
            added, deleted, changed = diff_size(parent_source, candidate_source)

            is_new_best = False
            is_accuracy_improvement = False
            is_tiebreak_improvement = False
            tiebreak_ce_improvement = math.nan
            if valid and math.isfinite(score) and score > best_score:
                is_new_best = True
                new_best_count += 1
                last_score_improvement = opportunity
                if math.isfinite(correct) and correct > best_correct:
                    is_accuracy_improvement = True
                    accuracy_improvement_count += 1
                    last_correct_improvement = opportunity
                    best_correct = int(correct)
                elif math.isfinite(correct) and int(correct) == int(best_correct):
                    is_tiebreak_improvement = True
                    tiebreak_improvement_count += 1
                    if math.isfinite(ce):
                        tiebreak_ce_improvement = best_ce - ce
                best_score = score
                if math.isfinite(ce):
                    best_ce = ce

            text_blob = " ".join([mechanism, hypothesis, intended_edit, evidence, message])
            numeric_evidence = bool(NUMBER_RE.search(evidence))
            evidence_update = bool(EVIDENCE_UPDATE_RE.search(evidence))
            preserve_ce = bool(PRESERVE_PREDICTIONS_RE.search(hypothesis) and LOWER_CE_RE.search(hypothesis))
            calibration = bool(CALIBRATION_RE.search(text_blob))
            tokens = usage_total(event.get("usage_increment"))
            evaluator_seconds = safe_float(event.get("evaluator_seconds_increment"))
            total_tokens += tokens
            if math.isfinite(evaluator_seconds):
                total_evaluator_seconds += evaluator_seconds

            proposal_rows.append({
                "run_id": run.run_id,
                "block": run.block,
                "scope": "original" if run.block <= 3 else "extension",
                "condition": run.condition,
                "opportunity": opportunity,
                "proposal_type": event.get("proposal_type"),
                "valid": int(valid),
                "failure_kind": evaluation.get("failure_kind") or "",
                "retained": int(bool(event.get("retained"))),
                "retention_decision": event.get("retention_decision") or "",
                "score": score,
                "correct": correct,
                "cross_entropy": ce,
                "parameters": metrics.get("parameters", ""),
                "is_new_best": int(is_new_best),
                "is_accuracy_improvement": int(is_accuracy_improvement),
                "is_tiebreak_improvement": int(is_tiebreak_improvement),
                "tiebreak_ce_improvement": tiebreak_ce_improvement,
                "best_score_after": best_score,
                "best_correct_after": best_correct,
                "declared_novelty": novelty,
                "source_available": int(source_available),
                "constant_only_edit": int(constant_only),
                "no_source_change": int(no_source_change),
                "structural_novelty": structural_novelty,
                "changed_lines": changed,
                "added_lines": added,
                "deleted_lines": deleted,
                "ast_signature": ast_signature(candidate_source) if candidate_source else "",
                "numeric_evidence": int(numeric_evidence),
                "evidence_update": int(evidence_update),
                "preserve_predictions_lower_ce": int(preserve_ce),
                "calibration_language": int(calibration),
                "usage_tokens": tokens,
                "evaluator_seconds": evaluator_seconds,
                "mechanism": mechanism,
                "hypothesis": hypothesis,
                "intended_edit": intended_edit,
                "evidence": evidence,
                "agent_message": message,
                "candidate_id": event.get("candidate_id") or "",
                "parent_id": parent_id or "",
            })

        run_proposals = [row for row in proposal_rows if row["run_id"] == run.run_id]
        after_plateau = [
            row for row in run_proposals
            if int(row["opportunity"]) > last_correct_improvement
        ]
        before_or_at_plateau = [
            row for row in run_proposals
            if int(row["opportunity"]) <= last_correct_improvement
        ]
        structural = [row for row in run_proposals if int(row["source_available"])]
        after_tokens = sum(int(row["usage_tokens"]) for row in after_plateau)
        after_eval = sum(
            safe_float(row["evaluator_seconds"])
            for row in after_plateau if math.isfinite(safe_float(row["evaluator_seconds"]))
        )
        final_metrics = final_candidate["metrics"]
        run_row = {
            "run_id": run.run_id,
            "block": run.block,
            "scope": "original" if run.block <= 3 else "extension",
            "condition": run.condition,
            "baseline_score": baseline_score,
            "baseline_correct": baseline_correct,
            "baseline_cross_entropy": baseline["validation_cross_entropy"],
            "final_score": final_metrics["validation_score"],
            "final_correct": final_metrics["validation_correct"],
            "final_cross_entropy": final_metrics["validation_cross_entropy"],
            "final_parameters": final_metrics["parameters"],
            "score_gain": float(final_metrics["validation_score"]) - baseline_score,
            "correct_gain": int(final_metrics["validation_correct"]) - baseline_correct,
            "cross_entropy_reduction": float(baseline["validation_cross_entropy"]) - float(final_metrics["validation_cross_entropy"]),
            "valid_count": sum(int(row["valid"]) for row in run_proposals),
            "valid_rate": statistics.fmean(int(row["valid"]) for row in run_proposals),
            "valid_rate_pre_plateau": statistics.fmean(int(row["valid"]) for row in before_or_at_plateau),
            "valid_rate_post_plateau": statistics.fmean(int(row["valid"]) for row in after_plateau) if after_plateau else math.nan,
            "retained_count": sum(int(row["retained"]) for row in run_proposals),
            "retained_rate_pre_plateau": statistics.fmean(int(row["retained"]) for row in before_or_at_plateau),
            "retained_rate_post_plateau": statistics.fmean(int(row["retained"]) for row in after_plateau) if after_plateau else math.nan,
            "new_best_count": new_best_count,
            "accuracy_improvement_count": accuracy_improvement_count,
            "tiebreak_improvement_count": tiebreak_improvement_count,
            "tiebreak_share_of_best_improvements": tiebreak_improvement_count / new_best_count if new_best_count else 0.0,
            "last_score_improvement": last_score_improvement,
            "last_accuracy_improvement": last_correct_improvement,
            "post_score_plateau_proposals": 200 - last_score_improvement,
            "post_accuracy_plateau_proposals": 200 - last_correct_improvement,
            "post_accuracy_plateau_fraction": (200 - last_correct_improvement) / 200,
            "total_tokens": total_tokens,
            "post_accuracy_plateau_tokens": after_tokens,
            "post_accuracy_plateau_token_fraction": after_tokens / total_tokens if total_tokens else 0.0,
            "total_evaluator_seconds": total_evaluator_seconds,
            "post_accuracy_plateau_evaluator_seconds": after_eval,
            "post_accuracy_plateau_evaluator_fraction": after_eval / total_evaluator_seconds if total_evaluator_seconds else 0.0,
            "constant_only_count": sum(int(row["constant_only_edit"]) for row in structural),
            "constant_only_rate": statistics.fmean(int(row["constant_only_edit"]) for row in structural) if structural else math.nan,
            "constant_only_rate_pre_plateau": statistics.fmean(int(row["constant_only_edit"]) for row in before_or_at_plateau if int(row["source_available"])) if any(int(row["source_available"]) for row in before_or_at_plateau) else math.nan,
            "constant_only_rate_post_plateau": statistics.fmean(int(row["constant_only_edit"]) for row in after_plateau if int(row["source_available"])) if any(int(row["source_available"]) for row in after_plateau) else math.nan,
            "mean_declared_novelty": statistics.fmean(float(row["declared_novelty"]) for row in run_proposals),
            "mean_declared_novelty_pre_plateau": statistics.fmean(float(row["declared_novelty"]) for row in before_or_at_plateau),
            "mean_declared_novelty_post_plateau": statistics.fmean(float(row["declared_novelty"]) for row in after_plateau) if after_plateau else math.nan,
            "mean_structural_novelty": statistics.fmean(float(row["structural_novelty"]) for row in structural),
            "mean_structural_novelty_pre_plateau": statistics.fmean(float(row["structural_novelty"]) for row in before_or_at_plateau if int(row["source_available"])) if any(int(row["source_available"]) for row in before_or_at_plateau) else math.nan,
            "mean_structural_novelty_post_plateau": statistics.fmean(float(row["structural_novelty"]) for row in after_plateau if int(row["source_available"])) if any(int(row["source_available"]) for row in after_plateau) else math.nan,
            "numeric_evidence_rate": statistics.fmean(int(row["numeric_evidence"]) for row in run_proposals),
            "evidence_update_rate": statistics.fmean(int(row["evidence_update"]) for row in run_proposals),
            "preserve_ce_rate": statistics.fmean(int(row["preserve_predictions_lower_ce"]) for row in run_proposals),
            "preserve_ce_rate_pre_plateau": statistics.fmean(int(row["preserve_predictions_lower_ce"]) for row in before_or_at_plateau),
            "preserve_ce_rate_post_plateau": statistics.fmean(int(row["preserve_predictions_lower_ce"]) for row in after_plateau) if after_plateau else math.nan,
            "calibration_language_rate": statistics.fmean(int(row["calibration_language"]) for row in run_proposals),
            "calibration_language_rate_pre_plateau": statistics.fmean(int(row["calibration_language"]) for row in before_or_at_plateau),
            "calibration_language_rate_post_plateau": statistics.fmean(int(row["calibration_language"]) for row in after_plateau) if after_plateau else math.nan,
            "tiebreak_improvement_rate_pre_plateau": statistics.fmean(int(row["is_tiebreak_improvement"]) for row in before_or_at_plateau),
            "tiebreak_improvement_rate_post_plateau": statistics.fmean(int(row["is_tiebreak_improvement"]) for row in after_plateau) if after_plateau else math.nan,
            "unique_declared_mechanisms": len({str(row["mechanism"]).strip().lower() for row in run_proposals}),
            "unique_ast_signatures": len({row["ast_signature"] for row in structural if row["ast_signature"]}),
        }
        run_rows.append(run_row)

        key_opportunities = {
            1, 2, 5, 10, 20, 50, 100, 150, 190, 200,
            last_correct_improvement,
            max(1, last_correct_improvement - 1),
            min(200, last_correct_improvement + 1),
            last_score_improvement,
        }
        tie_rows = [row for row in run_proposals if int(row["is_tiebreak_improvement"])]
        preserve_rows = [row for row in run_proposals if int(row["preserve_predictions_lower_ce"])]
        structural_rows = sorted(
            (row for row in run_proposals if int(row["source_available"])),
            key=lambda row: float(row["structural_novelty"]),
            reverse=True,
        )
        invalid_rows = [row for row in run_proposals if not int(row["valid"])]
        for collection in (tie_rows, preserve_rows, invalid_rows):
            if collection:
                key_opportunities.add(int(collection[0]["opportunity"]))
                key_opportunities.add(int(collection[len(collection) // 2]["opportunity"]))
                key_opportunities.add(int(collection[-1]["opportunity"]))
        for row in structural_rows[:2]:
            key_opportunities.add(int(row["opportunity"]))
        for row in run_proposals:
            if int(row["opportunity"]) in key_opportunities:
                qualitative_rows.append({
                    key: row[key] for key in [
                        "run_id", "block", "scope", "condition", "opportunity",
                        "valid", "retained", "score", "correct", "cross_entropy",
                        "is_new_best", "is_accuracy_improvement", "is_tiebreak_improvement",
                        "constant_only_edit", "structural_novelty", "changed_lines",
                        "preserve_predictions_lower_ce", "calibration_language",
                        "mechanism", "hypothesis", "intended_edit", "evidence", "agent_message",
                    ]
                })

    if len(proposal_rows) != 4000:
        raise ValueError(f"Expected 4,000 proposal rows, got {len(proposal_rows)}")

    last_accuracy_by_run = {
        row["run_id"]: int(row["last_accuracy_improvement"]) for row in run_rows
    }
    for row in proposal_rows:
        row["after_last_accuracy_improvement"] = int(
            int(row["opportunity"]) > last_accuracy_by_run[row["run_id"]]
        )

    phase_rows: list[dict[str, Any]] = []
    for start in range(1, 201, 20):
        end = start + 19
        group = [row for row in proposal_rows if start <= int(row["opportunity"]) <= end]
        source_group = [row for row in group if int(row["source_available"])]
        phase_rows.append({
            "opportunity_start": start,
            "opportunity_end": end,
            "n_proposals": len(group),
            "valid_rate": statistics.fmean(int(row["valid"]) for row in group),
            "retained_rate": statistics.fmean(int(row["retained"]) for row in group),
            "new_best_rate": statistics.fmean(int(row["is_new_best"]) for row in group),
            "accuracy_improvement_rate": statistics.fmean(int(row["is_accuracy_improvement"]) for row in group),
            "tiebreak_improvement_rate": statistics.fmean(int(row["is_tiebreak_improvement"]) for row in group),
            "literal_only_edit_rate": statistics.fmean(int(row["constant_only_edit"]) for row in source_group),
            "mean_source_structural_novelty": statistics.fmean(float(row["structural_novelty"]) for row in source_group),
            "mean_declared_novelty": statistics.fmean(float(row["declared_novelty"]) for row in group),
            "numeric_evidence_rate": statistics.fmean(int(row["numeric_evidence"]) for row in group),
            "evidence_update_rate": statistics.fmean(int(row["evidence_update"]) for row in group),
            "preserve_predictions_lower_ce_rate": statistics.fmean(int(row["preserve_predictions_lower_ce"]) for row in group),
            "calibration_language_rate": statistics.fmean(int(row["calibration_language"]) for row in group),
            "after_last_accuracy_improvement_rate": statistics.fmean(int(row["after_last_accuracy_improvement"]) for row in group),
            "tokens": sum(int(row["usage_tokens"]) for row in group),
            "evaluator_seconds": sum(float(row["evaluator_seconds"]) for row in group if math.isfinite(safe_float(row["evaluator_seconds"]))),
        })

    condition_rows = summarize_condition(run_rows)
    paired_rows: list[dict[str, Any]] = []
    for label, pre_key, post_key in [
        ("valid_rate", "valid_rate_pre_plateau", "valid_rate_post_plateau"),
        ("retained_rate", "retained_rate_pre_plateau", "retained_rate_post_plateau"),
        ("literal_only_edit_rate", "constant_only_rate_pre_plateau", "constant_only_rate_post_plateau"),
        ("source_structural_novelty", "mean_structural_novelty_pre_plateau", "mean_structural_novelty_post_plateau"),
        ("declared_novelty", "mean_declared_novelty_pre_plateau", "mean_declared_novelty_post_plateau"),
        ("preserve_predictions_lower_ce_rate", "preserve_ce_rate_pre_plateau", "preserve_ce_rate_post_plateau"),
        ("calibration_language_rate", "calibration_language_rate_pre_plateau", "calibration_language_rate_post_plateau"),
        ("tiebreak_improvement_rate", "tiebreak_improvement_rate_pre_plateau", "tiebreak_improvement_rate_post_plateau"),
    ]:
        usable = [
            row for row in run_rows
            if math.isfinite(safe_float(row[pre_key])) and math.isfinite(safe_float(row[post_key]))
        ]
        differences = [safe_float(row[post_key]) - safe_float(row[pre_key]) for row in usable]
        lo, hi = block_bootstrap_difference_ci(usable, pre_key, post_key)
        paired_rows.append({
            "measure": label,
            "n_runs": len(usable),
            "pre_mean": statistics.fmean(safe_float(row[pre_key]) for row in usable),
            "post_mean": statistics.fmean(safe_float(row[post_key]) for row in usable),
            "mean_paired_difference": statistics.fmean(differences),
            "median_paired_difference": statistics.median(differences),
            "block_bootstrap_ci_low": lo,
            "block_bootstrap_ci_high": hi,
            "runs_increasing": sum(value > 0 for value in differences),
            "runs_decreasing": sum(value < 0 for value in differences),
            "runs_unchanged": sum(value == 0 for value in differences),
        })
    correlations: list[dict[str, Any]] = []
    for process_key in [
        "valid_rate", "new_best_count", "accuracy_improvement_count",
        "tiebreak_improvement_count", "post_accuracy_plateau_fraction",
        "post_accuracy_plateau_token_fraction", "constant_only_rate",
        "mean_declared_novelty", "mean_structural_novelty", "numeric_evidence_rate",
        "evidence_update_rate", "preserve_ce_rate", "calibration_language_rate",
        "unique_declared_mechanisms", "unique_ast_signatures", "total_tokens",
        "total_evaluator_seconds",
    ]:
        rho = spearman(
            [safe_float(row["correct_gain"]) for row in run_rows],
            [safe_float(row[process_key]) for row in run_rows],
        )
        lo, hi = block_bootstrap_ci(run_rows, "correct_gain", process_key)
        original = [row for row in run_rows if row["scope"] == "original"]
        extension = [row for row in run_rows if row["scope"] == "extension"]
        correlations.append({
            "outcome": "correct_gain",
            "process_measure": process_key,
            "spearman_all": rho,
            "block_bootstrap_ci_low": lo,
            "block_bootstrap_ci_high": hi,
            "spearman_original_blocks_1_3": spearman(
                [safe_float(row["correct_gain"]) for row in original],
                [safe_float(row[process_key]) for row in original],
            ),
            "spearman_extension_blocks_4_5": spearman(
                [safe_float(row["correct_gain"]) for row in extension],
                [safe_float(row[process_key]) for row in extension],
            ),
        })

    failure_counts = Counter(
        str(row["failure_kind"] or "valid") for row in proposal_rows
    )
    source_rows = [row for row in proposal_rows if int(row["source_available"])]
    literal_rows = [row for row in source_rows if int(row["constant_only_edit"])]
    structural_rows = [row for row in source_rows if not int(row["constant_only_edit"])]
    edit_class_rows: list[dict[str, Any]] = []
    for label, group in [("literal_only", literal_rows), ("structural_token_change", structural_rows)]:
        edit_class_rows.append({
            "edit_class": label,
            "n": len(group),
            "valid_rate": statistics.fmean(int(row["valid"]) for row in group),
            "timeout_rate": statistics.fmean(row["failure_kind"] == "timeout" for row in group),
            "retained_rate": statistics.fmean(int(row["retained"]) for row in group),
            "accuracy_improvement_rate": statistics.fmean(int(row["is_accuracy_improvement"]) for row in group),
            "tiebreak_improvement_rate": statistics.fmean(int(row["is_tiebreak_improvement"]) for row in group),
            "mean_changed_lines": statistics.fmean(int(row["changed_lines"]) for row in group),
            "mean_structural_novelty": statistics.fmean(float(row["structural_novelty"]) for row in group),
        })
    tiebreak_ce_values = np.asarray([
        float(row["tiebreak_ce_improvement"])
        for row in proposal_rows
        if int(row["is_tiebreak_improvement"])
    ])
    if len(tiebreak_ce_values) != 762 or not np.all(tiebreak_ce_values > 0):
        raise ValueError("Unexpected same-count cross-entropy improvement distribution")
    tiebreak_magnitude_rows: list[dict[str, Any]] = []
    for label, quantile in [
        ("minimum", 0.0), ("p10", 0.10), ("p25", 0.25),
        ("median", 0.50), ("p75", 0.75), ("p90", 0.90),
        ("p95", 0.95), ("p99", 0.99), ("maximum", 1.0),
    ]:
        tiebreak_magnitude_rows.append({
            "statistic": label,
            "cross_entropy_improvement": float(np.quantile(tiebreak_ce_values, quantile)),
            "fraction_at_or_below": quantile,
        })
    for threshold in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        tiebreak_magnitude_rows.append({
            "statistic": f"at_or_below_{threshold:.0e}",
            "cross_entropy_improvement": threshold,
            "fraction_at_or_below": float(np.mean(tiebreak_ce_values <= threshold)),
        })
    scope_rows: list[dict[str, Any]] = []
    for scope in ["original", "extension", "all"]:
        scoped_proposals = proposal_rows if scope == "all" else [row for row in proposal_rows if row["scope"] == scope]
        scoped_runs = run_rows if scope == "all" else [row for row in run_rows if row["scope"] == scope]
        scoped_source = [row for row in scoped_proposals if int(row["source_available"])]
        scoped_pre = [row for row in scoped_source if not int(row["after_last_accuracy_improvement"])]
        scoped_post = [row for row in scoped_source if int(row["after_last_accuracy_improvement"])]
        scoped_new_best = sum(int(row["is_new_best"]) for row in scoped_proposals)
        scoped_total_tokens = sum(int(row["usage_tokens"]) for row in scoped_proposals)
        scoped_total_eval = sum(float(row["evaluator_seconds"]) for row in scoped_proposals if math.isfinite(safe_float(row["evaluator_seconds"])))
        scope_rows.append({
            "scope": scope,
            "n_runs": len(scoped_runs),
            "n_proposals": len(scoped_proposals),
            "valid_rate": statistics.fmean(int(row["valid"]) for row in scoped_proposals),
            "retained_rate": statistics.fmean(int(row["retained"]) for row in scoped_proposals),
            "new_best_events": scoped_new_best,
            "accuracy_improvement_events": sum(int(row["is_accuracy_improvement"]) for row in scoped_proposals),
            "tiebreak_improvement_events": sum(int(row["is_tiebreak_improvement"]) for row in scoped_proposals),
            "tiebreak_share_of_new_best": sum(int(row["is_tiebreak_improvement"]) for row in scoped_proposals) / scoped_new_best,
            "post_accuracy_plateau_proposal_fraction": statistics.fmean(int(row["after_last_accuracy_improvement"]) for row in scoped_proposals),
            "post_accuracy_plateau_token_fraction": sum(int(row["usage_tokens"]) for row in scoped_proposals if int(row["after_last_accuracy_improvement"])) / scoped_total_tokens,
            "post_accuracy_plateau_evaluator_fraction": sum(float(row["evaluator_seconds"]) for row in scoped_proposals if int(row["after_last_accuracy_improvement"]) and math.isfinite(safe_float(row["evaluator_seconds"]))) / scoped_total_eval,
            "literal_only_edit_rate_pre_plateau": statistics.fmean(int(row["constant_only_edit"]) for row in scoped_pre),
            "literal_only_edit_rate_post_plateau": statistics.fmean(int(row["constant_only_edit"]) for row in scoped_post),
            "structural_novelty_pre_plateau": statistics.fmean(float(row["structural_novelty"]) for row in scoped_pre),
            "structural_novelty_post_plateau": statistics.fmean(float(row["structural_novelty"]) for row in scoped_post),
            "mean_correct_gain": statistics.fmean(float(row["correct_gain"]) for row in scoped_runs),
        })
    aggregate = {
        "campaign": str(campaign.relative_to(REPO)) if campaign.is_relative_to(REPO) else campaign.name,
        "runs": len(run_rows),
        "proposals": len(proposal_rows),
        "valid_proposals": sum(int(row["valid"]) for row in proposal_rows),
        "invalid_proposals": sum(1 - int(row["valid"]) for row in proposal_rows),
        "retained_proposals": sum(int(row["retained"]) for row in proposal_rows),
        "new_best_events": sum(int(row["is_new_best"]) for row in proposal_rows),
        "accuracy_improvement_events": sum(int(row["is_accuracy_improvement"]) for row in proposal_rows),
        "tiebreak_improvement_events": sum(int(row["is_tiebreak_improvement"]) for row in proposal_rows),
        "raw_agent_messages": sum(bool(str(row["agent_message"] or "").strip()) for row in proposal_rows),
        "qualitative_selected_rows": len(qualitative_rows),
        "qualitative_selected_raw_messages": sum(bool(str(row["agent_message"] or "").strip()) for row in qualitative_rows),
        "tiebreak_ce_improvement_median": float(np.median(tiebreak_ce_values)),
        "tiebreak_ce_improvement_p90": float(np.quantile(tiebreak_ce_values, 0.90)),
        "tiebreak_ce_improvement_fraction_at_or_below_1e_6": float(np.mean(tiebreak_ce_values <= 1e-6)),
        "tiebreak_ce_improvement_fraction_at_or_below_1e_4": float(np.mean(tiebreak_ce_values <= 1e-4)),
        "tiebreak_ce_improvement_fraction_above_1e_3": float(np.mean(tiebreak_ce_values > 1e-3)),
        "total_tokens": sum(int(row["usage_tokens"]) for row in proposal_rows),
        "total_evaluator_seconds": sum(safe_float(row["evaluator_seconds"]) for row in proposal_rows if math.isfinite(safe_float(row["evaluator_seconds"]))),
        "post_accuracy_plateau_proposals": sum(int(row["post_accuracy_plateau_proposals"]) for row in run_rows),
        "post_accuracy_plateau_tokens": sum(int(row["post_accuracy_plateau_tokens"]) for row in run_rows),
        "post_accuracy_plateau_evaluator_seconds": sum(float(row["post_accuracy_plateau_evaluator_seconds"]) for row in run_rows),
        "post_accuracy_plateau_proposal_fraction": sum(int(row["post_accuracy_plateau_proposals"]) for row in run_rows) / len(proposal_rows),
        "post_accuracy_plateau_token_fraction": sum(int(row["post_accuracy_plateau_tokens"]) for row in run_rows) / sum(int(row["usage_tokens"]) for row in proposal_rows),
        "post_accuracy_plateau_evaluator_fraction": sum(float(row["post_accuracy_plateau_evaluator_seconds"]) for row in run_rows) / sum(safe_float(row["evaluator_seconds"]) for row in proposal_rows if math.isfinite(safe_float(row["evaluator_seconds"]))),
        "tiebreak_share_of_new_best_events": sum(int(row["is_tiebreak_improvement"]) for row in proposal_rows) / sum(int(row["is_new_best"]) for row in proposal_rows),
        "literal_only_edit_rate_pre_plateau": statistics.fmean(
            int(row["constant_only_edit"]) for row in proposal_rows
            if int(row["source_available"]) and not int(row["after_last_accuracy_improvement"])
        ),
        "literal_only_edit_rate_post_plateau": statistics.fmean(
            int(row["constant_only_edit"]) for row in proposal_rows
            if int(row["source_available"]) and int(row["after_last_accuracy_improvement"])
        ),
        "structural_novelty_pre_plateau": statistics.fmean(
            float(row["structural_novelty"]) for row in proposal_rows
            if int(row["source_available"]) and not int(row["after_last_accuracy_improvement"])
        ),
        "structural_novelty_post_plateau": statistics.fmean(
            float(row["structural_novelty"]) for row in proposal_rows
            if int(row["source_available"]) and int(row["after_last_accuracy_improvement"])
        ),
        "source_snapshots_available": len(source_rows),
        "edit_class_summary": {row["edit_class"]: row for row in edit_class_rows},
        "failure_counts": dict(sorted(failure_counts.items())),
        "score_definition": "validation_correct + 0.5 / (1 + validation_cross_entropy)",
        "original_scope_runs": 12,
        "extension_scope_runs": 8,
    }

    write_csv(output / "proposal_table.csv", proposal_rows)
    write_csv(output / "run_summary.csv", run_rows)
    write_csv(output / "condition_summary.csv", condition_rows)
    write_csv(output / "correlations.csv", correlations)
    write_csv(output / "phase_summary.csv", phase_rows)
    write_csv(output / "edit_class_summary.csv", edit_class_rows)
    write_csv(output / "tiebreak_magnitude_summary.csv", tiebreak_magnitude_rows)
    write_csv(output / "scope_summary.csv", scope_rows)
    write_csv(output / "paired_pre_post_summary.csv", paired_rows)
    write_csv(output / "qualitative_sample_manifest.csv", qualitative_rows)
    with (output / "aggregate.json").open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2, sort_keys=True)
        fh.write("\n")

    make_figures(proposal_rows, run_rows, output)
    write_qualitative_reader(qualitative_rows, output / "qualitative_reader.md")
    return aggregate


def make_figures(
    proposals: Sequence[dict[str, Any]], runs: Sequence[dict[str, Any]], output: Path
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
    })
    colors = {"C0": "#3b82f6", "C1": "#f59e0b", "C2": "#8b5cf6", "C3": "#16a34a"}

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
    ordered = sorted(runs, key=lambda row: (int(row["last_accuracy_improvement"]), row["run_id"]))
    y = np.arange(len(ordered))
    for index, row in enumerate(ordered):
        color = colors[row["condition"]]
        axes[0].hlines(index, int(row["last_accuracy_improvement"]), 200, color=color, linewidth=3)
        axes[0].plot(int(row["last_accuracy_improvement"]), index, "o", color=color, ms=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"B{row['block']}{row['condition']}" for row in ordered])
    axes[0].set_xlim(0, 200)
    axes[0].set_xlabel("Proposal opportunity")
    axes[0].set_title("Search remaining after last accuracy gain")
    axes[0].grid(axis="x", alpha=0.22)

    xs = [float(row["correct_gain"]) for row in runs]
    ys = [float(row["post_accuracy_plateau_token_fraction"]) for row in runs]
    for row, x, yval in zip(runs, xs, ys, strict=True):
        axes[1].scatter(x, yval, color=colors[row["condition"]], s=30, edgecolor="white", linewidth=0.5)
        axes[1].annotate(f"B{row['block']}{row['condition']}", (x, yval), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
    axes[1].set_xlabel("Validation-correct gain over baseline")
    axes[1].set_ylabel("Fraction of agent tokens after last accuracy gain")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Endpoint gain does not reveal post-plateau effort")
    axes[1].grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output / "fig1_plateau_audit.pdf")
    fig.savefig(output / "fig1_plateau_audit.png")
    plt.close(fig)

    bins = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100),
            (101, 120), (121, 140), (141, 160), (161, 180), (181, 200)]
    novelty = []
    constant = []
    preserve = []
    for lo, hi in bins:
        group = [row for row in proposals if lo <= int(row["opportunity"]) <= hi]
        source_group = [row for row in group if int(row["source_available"])]
        novelty.append(statistics.fmean(float(row["structural_novelty"]) for row in source_group))
        constant.append(statistics.fmean(int(row["constant_only_edit"]) for row in source_group))
        preserve.append(statistics.fmean(int(row["preserve_predictions_lower_ce"]) for row in group))
    centers = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(7.1, 3.05))
    ax.plot(centers, novelty, marker="o", label="Mean source-structural novelty", color="#2563eb")
    ax.plot(centers, constant, marker="s", label="Literal-only edit rate", color="#dc2626")
    ax.plot(centers, preserve, marker="^", label="Explicit preserve-predictions/lower-CE rate", color="#059669")
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in bins], rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Proposal-opportunity bin")
    ax.set_ylabel("Rate / normalized distance")
    ax.set_title("Search behavior changes across the 200-proposal horizon")
    ax.grid(alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(output / "fig2_search_dynamics.pdf")
    fig.savefig(output / "fig2_search_dynamics.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 2.55))
    labels = [f"B{row['block']}{row['condition']}" for row in runs]
    acc = [int(row["accuracy_improvement_count"]) for row in runs]
    tie = [int(row["tiebreak_improvement_count"]) for row in runs]
    x = np.arange(len(runs))
    ax.bar(x, acc, color="#2563eb", label="New-best events increasing correct count")
    ax.bar(x, tie, bottom=acc, color="#f97316", label="New-best events changing only tie-break score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel("Number of new-best events")
    ax.set_title("A scalar leaderboard conflates accuracy gains with tie-break optimization")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "fig3_improvement_types.pdf")
    fig.savefig(output / "fig3_improvement_types.png")
    plt.close(fig)


def write_qualitative_reader(rows: Sequence[dict[str, Any]], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row["block"], row["condition"], row["opportunity"]))
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Systematic qualitative reader\n\n")
        fh.write(
            "This file contains a deterministic, rule-selected set of opportunities for "
            "every trajectory: early and late search, fixed checkpoints, the final "
            "correct-count improvement and its neighbors, representative first/middle/last "
            "same-count score improvements and invalid events, and the two largest source "
            "changes. Selection is for audit coverage, not statistical generalization.\n\n"
        )
        for row in ordered:
            fh.write(
                f"## B{row['block']}{row['condition']} opportunity {row['opportunity']}\n\n"
                f"- valid={row['valid']}; retained={row['retained']}; correct={row['correct']}; "
                f"score={row['score']}; CE={row['cross_entropy']}\n"
                f"- accuracy_improvement={row['is_accuracy_improvement']}; "
                f"tie_break_improvement={row['is_tiebreak_improvement']}; "
                f"constant_only={row['constant_only_edit']}; structural_novelty={row['structural_novelty']}\n"
                f"- mechanism: {row['mechanism']}\n"
                f"- hypothesis: {row['hypothesis']}\n"
                f"- intended edit: {row['intended_edit']}\n"
                f"- evidence: {row['evidence']}\n\n"
            )
            message = str(row["agent_message"] or "").strip()
            if message:
                fh.write("```text\n" + message[:5000] + "\n```\n\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    aggregate = analyse(args.campaign.resolve(), args.output.resolve())
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
