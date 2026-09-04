#!/usr/bin/env python3
# ruff: noqa: E501
"""Reproduce Paper 2's state-matched prompt-intervention analysis.

The primary corpus is the pair of unified-v3 Tiny AdderBoard campaigns.  In
each block C0/C1 and C2/C3 are literal shared trajectories through opportunity
9, then fork from the same state at opportunity 10.  C1/C3 receive an explicit
assumption-challenge direction at every tenth opportunity; C0/C2 receive the
ordinary direction.  All primary analyses stop at opportunity 70, the largest
common horizon available for every trajectory when this analysis was frozen.

The completed Fashion-MNIST v2.1 campaign is used only as a cross-task
descriptive replication because its same-seed trajectories had already
diverged before their intervention checkpoints.
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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_OUTPUT = HERE / "derived"
GREEDY = REPO / "data/c0c3/unified-v3-tiny-adderboard-greedy-campaign"
NATIVE = REPO / "data/c0c3/unified-v3-tiny-adderboard-native-campaign"
FASHION = REPO / "data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign"
COMMON_HORIZON = 70
FORK = 10
SEED = 20260901

WORD_RE = re.compile(r"[a-z][a-z0-9_+-]{1,}")
NUMBER_RE = re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", re.I
)
ASSUMPTION_RE = re.compile(
    r"\b(?:assum(?:e|ed|es|ing|ption|ptions)|load[- ]bearing|challenge|alternative|"
    r"different (?:mechanism|architecture|representation|computation)|falsif(?:y|ied|iable|ication))\b",
    re.I,
)
MECHANISM_SHIFT_RE = re.compile(
    r"\b(?:instead|rather than|replace|factor|decouple|separate|parallel|hierarch|"
    r"different|alternative|new representation|new mechanism|reframe|challenge)\b",
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
    "proposal", "opportunity", "transformer", "learned", "learning",
}

FAMILY_PATTERNS: dict[str, re.Pattern[str]] = {
    "width_capacity": re.compile(r"\b(width|dimension|dimensional|heads?|layers?|feed[- ]?forward|ffn|capacity|bottleneck)\b", re.I),
    "token_embedding": re.compile(r"\b(token|symbol|embedding|vocab|codebook|lookup|manifold)\b", re.I),
    "position_carry": re.compile(r"\b(position|positional|carry|digit|column|place value|convolution|local)\b", re.I),
    "attention_qkv": re.compile(r"\b(attention|qkv|query|key|value projection|head mixing|attn)\b", re.I),
    "ffn_nonlinearity": re.compile(r"\b(feed[- ]?forward|ffn|mlp|nonlinear|activation|gating|gate)\b", re.I),
    "tying_factorization": re.compile(r"\b(tie|tying|factor|factoriz|low[- ]rank|shared weight|packed|coefficient|scalar|prun|fixed value)\b", re.I),
    "optimization": re.compile(r"\b(optimizer|learning rate|schedule|warmup|cosine|batch|gradient|decay|training trajectory|initializ|seed)\b", re.I),
    "readout_output": re.compile(r"\b(readout|output head|classifier|logit|decode|decoder|projection head)\b", re.I),
    "regularization": re.compile(r"\b(dropout|regulariz|noise|augment|ensemble|averag|ema)\b", re.I),
}


@dataclass(frozen=True)
class Run:
    architecture: str
    campaign: Path
    run_dir: Path
    run_id: str
    block: int
    condition: str
    memory: str
    treated: bool
    baseline: dict[str, Any]
    state: dict[str, Any]
    events: tuple[dict[str, Any], ...]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    return rows


def load_runs(campaign: Path, architecture: str) -> list[Run]:
    output: list[Run] = []
    for run_dir in sorted((campaign / "runs").iterdir()):
        if not run_dir.is_dir() or not (run_dir / "manifest.json").exists():
            continue
        manifest = load_json(run_dir / "manifest.json")
        assignment = manifest["assignment"]
        condition = assignment["condition"]
        events = tuple(
            row
            for row in load_jsonl(run_dir / "events.jsonl")
            if row.get("event") == "proposal_completed"
        )
        by_opportunity = {int(row["opportunity"]): row for row in events}
        if len(by_opportunity) != len(events):
            raise ValueError(f"{run_dir.name}: duplicate completed opportunities")
        output.append(
            Run(
                architecture=architecture,
                campaign=campaign,
                run_dir=run_dir,
                run_id=assignment["run_id"],
                block=int(assignment["block"]),
                condition=condition,
                memory="portfolio" if condition in {"C2", "C3"} else "single",
                treated=condition in {"C1", "C3"},
                baseline=manifest["baseline"],
                state=load_json(run_dir / "state.json"),
                events=tuple(sorted(events, key=lambda row: int(row["opportunity"]))),
            )
        )
    return output


def event_map(run: Run) -> dict[int, dict[str, Any]]:
    return {int(row["opportunity"]): row for row in run.events}


def words(text: str) -> set[str]:
    return {
        word
        for word in WORD_RE.findall((text or "").lower())
        if len(word) > 2 and word not in STOPWORDS
    }


def jaccard_distance(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def normalized_python_tokens(source: str) -> list[str]:
    output: list[str] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in stream:
            if token.type in {
                tokenize.ENCODING,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
                tokenize.COMMENT,
            }:
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
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return Counter()
    return Counter(type(node).__name__ for node in ast.walk(tree))


def counter_distance(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    total = sum(max(left[key], right[key]) for key in keys)
    if total == 0:
        return 0.0
    overlap = sum(min(left[key], right[key]) for key in keys)
    return 1.0 - overlap / total


def diff_counts(parent: str, candidate: str) -> tuple[int, int, int]:
    added = 0
    deleted = 0
    for line in difflib.unified_diff(parent.splitlines(), candidate.splitlines(), n=0):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            deleted += 1
    return added + deleted, added, deleted


def read_candidate_source(run: Run, candidate_id: str | None) -> str | None:
    if not candidate_id:
        return None
    path = run.run_dir / "candidates" / candidate_id / "train.py"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def event_text(event: dict[str, Any]) -> str:
    return " ".join(
        str(event.get(field) or "")
        for field in ("mechanism", "hypothesis", "intended_edit", "evidence")
    )


def family_tags(text: str) -> set[str]:
    return {name for name, pattern in FAMILY_PATTERNS.items() if pattern.search(text)}


def mechanism_family(text: str) -> str:
    """Assign one mutually exclusive, interpretable family to a proposal.

    The taxonomy was induced after reading all 64 fork summaries.  Ordering is
    intentional: a proposal that reuses attention projections as an FFN is
    coded as projection reuse rather than generic feedforward compression;
    likewise recurrent depth and token-interface factorization take priority
    over incidental width language.
    """
    lowered = text.lower()
    if (
        re.search(r"reus(?:e|ing).*(?:projection|mixer).*(?:feedforward|ffn)", lowered)
        or re.search(r"(?:feedforward|ffn).*(?:reus(?:e|ing)).*(?:projection|mixer)", lowered)
    ):
        return "projection_reuse"
    if re.search(r"relative[- ](?:offset|distance|position)|remove absolute positional", lowered):
        return "relative_position_attention"
    if re.search(
        r"recurrent|iterative|shared[- ]depth|depth[- ]for[- ]width|"
        r"appl(?:y|ication|ications).*(?:twice|two)|two full learned causal-attention blocks",
        lowered,
    ):
        return "iterative_or_shared_depth"
    if re.search(
        r"lexical|vocabulary|symbol manifold|symbol representation|token (?:code|codebook|interface)|"
        r"token/input-output|token embedding|embedding/output|tied embedding|rank[- ]?\d+.*(?:token|symbol)|"
        r"factor(?:ed|ing|iz(?:e|ing)).*(?:token|embedding)",
        lowered,
    ):
        return "token_interface_factorization"
    if re.search(
        r"multi[- ]query|shared[- ]key/value|shared key/value|key/value.*shar|shar(?:e|ing).*key/value|"
        r"narrow routing|routing subspace|attention communication bottleneck|"
        r"query/key heads?|qkv projection|replacing residual[- ]width attention",
        lowered,
    ):
        return "attention_routing_reparameterization"
    if re.search(r"layernorm|normalization|pre[- ]normal|bias|affine|offset[- ]free", lowered):
        return "normalization_or_bias_pruning"
    if re.search(r"feed[- ]?forward|ffn|nonlinear bottleneck|positionwise nonlinear", lowered):
        return "feedforward_compression"
    if re.search(r"(?:residual|model) width|attention heads?", lowered):
        return "width_or_head_adjustment"
    return "other"


def lexical_novelty(event: dict[str, Any], prior_events: Sequence[dict[str, Any]]) -> float:
    current = words(event_text(event))
    histories = [words(event_text(prior)) for prior in prior_events]
    if not histories:
        return 1.0
    return min(jaccard_distance(current, prior) for prior in histories)


def usage_total(usage: dict[str, Any] | None) -> int:
    usage = usage or {}
    if usage.get("total_tokens") is not None:
        return int(usage["total_tokens"] or 0)
    return int(usage.get("input_tokens", 0) or 0) + int(
        usage.get("output_tokens", 0) or 0
    )


def event_record(run: Run, event: dict[str, Any]) -> dict[str, Any]:
    opportunity = int(event["opportunity"])
    mapping = event_map(run)
    prior = [mapping[index] for index in range(1, opportunity) if index in mapping]
    parent_ids = event.get("selected_parent_ids") or event.get("parent_ids") or []
    parent_id = parent_ids[0] if parent_ids else None
    candidate_id = event.get("candidate_id")
    parent_source = read_candidate_source(run, parent_id)
    candidate_source = read_candidate_source(run, candidate_id)
    source_available = parent_source is not None and candidate_source is not None
    parent_tokens = normalized_python_tokens(parent_source or "")
    candidate_tokens = normalized_python_tokens(candidate_source or "")
    literal_only = bool(
        source_available
        and parent_source != candidate_source
        and parent_tokens == candidate_tokens
    )
    no_source_change = bool(source_available and parent_source == candidate_source)
    structural_novelty = (
        jaccard_distance(ngrams(parent_tokens), ngrams(candidate_tokens))
        if source_available
        else math.nan
    )
    ast_distance = (
        counter_distance(ast_nodes(parent_source), ast_nodes(candidate_source))
        if source_available
        else math.nan
    )
    changed, added, deleted = (
        diff_counts(parent_source, candidate_source)
        if source_available
        else (0, 0, 0)
    )
    evaluation = event.get("evaluation") or {}
    metrics = evaluation.get("metrics") or {}
    text = event_text(event)
    prior_tags: set[str] = set()
    for previous in prior:
        prior_tags.update(family_tags(event_text(previous)))
    tags = family_tags(text)
    new_family = bool(tags - prior_tags)
    output_tokens = int((event.get("usage_increment") or {}).get("output_tokens", 0) or 0)
    incumbent_before_id = (
        run.baseline.get("candidate_id")
        if opportunity == 1
        else mapping[opportunity - 1].get("incumbent_after")
    )
    incumbent_after_id = event.get("incumbent_after")
    incumbent_before_metrics = candidate_metrics(run, incumbent_before_id) or {}
    incumbent_after_metrics = candidate_metrics(run, incumbent_after_id) or {}
    incumbent_before_parameters = safe_float(incumbent_before_metrics.get("parameters"))
    incumbent_after_parameters = safe_float(incumbent_after_metrics.get("parameters"))
    immediate_parameter_reduction = (
        incumbent_before_parameters - incumbent_after_parameters
        if math.isfinite(incumbent_before_parameters)
        and math.isfinite(incumbent_after_parameters)
        else math.nan
    )
    return {
        "architecture": run.architecture,
        "run_id": run.run_id,
        "block": run.block,
        "condition": run.condition,
        "memory": run.memory,
        "treated": int(run.treated),
        "opportunity": opportunity,
        "transition": int(opportunity % 10 == 0),
        "proposal_type": event.get("proposal_type") or "",
        "candidate_id": candidate_id or "",
        "parent_id": parent_id or "",
        "source_available": int(source_available),
        "source_changed": int(source_available and not no_source_change),
        "literal_only": int(literal_only),
        "structural_edit": int(source_available and not no_source_change and not literal_only),
        "structural_novelty": structural_novelty,
        "ast_distance": ast_distance,
        "changed_lines": changed,
        "added_lines": added,
        "deleted_lines": deleted,
        "declared_lexical_novelty": lexical_novelty(event, prior),
        "family_tags": ";".join(sorted(tags)),
        # The mechanism field names the proposed computation.  Other fields
        # often mention preserved incumbent components or prior failures, which
        # would miscode a feedforward edit as (for example) token factorization.
        "mechanism_family": mechanism_family(str(event.get("mechanism") or text)),
        "new_family_tag": int(new_family),
        "assumption_language": int(bool(ASSUMPTION_RE.search(text))),
        "mechanism_shift_language": int(bool(MECHANISM_SHIFT_RE.search(text))),
        "numeric_evidence": int(bool(NUMBER_RE.search(str(event.get("evidence") or "")))),
        "qualified": int(bool(evaluation.get("valid"))),
        "failure_kind": evaluation.get("failure_kind") or "valid",
        "retained": int(bool(event.get("retained"))),
        "became_incumbent": int(
            bool(event.get("retained"))
            and event.get("candidate_id") == event.get("incumbent_after")
        ),
        "incumbent_parameters_before": incumbent_before_parameters,
        "incumbent_parameters_after": incumbent_after_parameters,
        "immediate_parameter_reduction": immediate_parameter_reduction,
        "accuracy": metrics.get("accuracy"),
        "parameters": metrics.get("parameters"),
        "training_steps": metrics.get("training_steps"),
        "evaluator_seconds": float(event.get("evaluator_seconds_increment") or 0.0),
        "total_tokens": usage_total(event.get("usage_increment")),
        "input_tokens": int((event.get("usage_increment") or {}).get("input_tokens", 0) or 0),
        "cached_input_tokens": int((event.get("usage_increment") or {}).get("cached_input_tokens", 0) or 0),
        "output_tokens": output_tokens,
        "mechanism": event.get("mechanism") or "",
        "hypothesis": event.get("hypothesis") or "",
        "intended_edit": event.get("intended_edit") or "",
        "evidence": event.get("evidence") or "",
        "incumbent_after": event.get("incumbent_after") or "",
    }


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


def incumbent_parameters_at(run: Run, opportunity: int) -> float:
    event = event_map(run)[opportunity]
    metrics = candidate_metrics(run, event.get("incumbent_after"))
    if not metrics or metrics.get("parameters") is None:
        raise ValueError(
            f"{run.run_id} opportunity {opportunity}: incumbent metrics unavailable"
        )
    return float(metrics["parameters"])


def validate_primary(runs: Sequence[Run]) -> dict[str, Any]:
    if len(runs) != 64:
        raise ValueError(f"Expected 64 Tiny AdderBoard runs, found {len(runs)}")
    for run in runs:
        opportunities = set(event_map(run))
        missing = set(range(1, COMMON_HORIZON + 1)) - opportunities
        if missing:
            raise ValueError(f"{run.run_id}: missing common-horizon rows {sorted(missing)}")

    report: dict[str, Any] = {
        "runs": len(runs),
        "common_horizon": COMMON_HORIZON,
        "pairs": 0,
        "prefix_event_matches": 0,
        "prefix_source_or_provenance_matches": 0,
        "fork_parent_matches": 0,
        "fork_prompt_insertion_only": 0,
    }
    grouped = {(run.architecture, run.block, run.condition): run for run in runs}
    for architecture in ("greedy", "native"):
        campaign = GREEDY if architecture == "greedy" else NATIVE
        prefix_events = load_jsonl(campaign / "paired-prefix-events.jsonl")
        if len(prefix_events) != 16 * 9:
            raise ValueError(f"{architecture}: expected 144 prefix mirror receipts")
        for block in range(1, 9):
            for control_condition, treated_condition in (("C0", "C1"), ("C2", "C3")):
                control = grouped[(architecture, block, control_condition)]
                treated = grouped[(architecture, block, treated_condition)]
                report["pairs"] += 1
                control_events = event_map(control)
                treated_events = event_map(treated)
                for opportunity in range(1, FORK):
                    left = control_events[opportunity]
                    right = treated_events[opportunity]
                    if left.get("candidate_id") != right.get("candidate_id"):
                        raise ValueError(
                            f"{architecture} B{block} {control_condition}/{treated_condition}: "
                            f"candidate mismatch at shared opportunity {opportunity}"
                        )
                    scientific_fields = (
                        "selected_parent_ids",
                        "parent_ids",
                        "evaluation",
                        "mechanism",
                        "hypothesis",
                        "intended_edit",
                        "evidence",
                        "retained",
                        "retention_decision",
                        "incumbent_after",
                        "portfolio_after",
                    )
                    if any(left.get(field) != right.get(field) for field in scientific_fields):
                        raise ValueError(
                            f"{architecture} B{block} {control_condition}/{treated_condition}: "
                            f"scientific event mismatch at shared opportunity {opportunity}"
                        )
                    report["prefix_event_matches"] += 1
                    left_source = read_candidate_source(control, left.get("candidate_id"))
                    right_source = read_candidate_source(treated, right.get("candidate_id"))
                    if left_source is not None and right_source is not None:
                        if left_source != right_source:
                            raise ValueError(
                                f"{architecture} B{block} {control_condition}/{treated_condition}: "
                                f"source mismatch at shared opportunity {opportunity}"
                            )
                    else:
                        left_provenance = load_json(
                            control.run_dir
                            / "opportunities"
                            / f"{opportunity:04d}"
                            / "candidate-provenance.json"
                        )
                        right_provenance = load_json(
                            treated.run_dir
                            / "opportunities"
                            / f"{opportunity:04d}"
                            / "candidate-provenance.json"
                        )
                        if (
                            left_provenance.get("diff_sha256")
                            != right_provenance.get("diff_sha256")
                            or left_provenance.get("semantic_delta_fingerprint")
                            != right_provenance.get("semantic_delta_fingerprint")
                        ):
                            raise ValueError(
                                f"{architecture} B{block} {control_condition}/{treated_condition}: "
                                f"provenance mismatch at shared opportunity {opportunity}"
                            )
                    report["prefix_source_or_provenance_matches"] += 1
                if (
                    control_events[FORK].get("selected_parent_ids")
                    != treated_events[FORK].get("selected_parent_ids")
                ):
                    raise ValueError(
                        f"{architecture} B{block} {control_condition}/{treated_condition}: "
                        "fork parents differ"
                    )
                report["fork_parent_matches"] += 1
                control_prompt = (
                    control.run_dir / "opportunities" / f"{FORK:04d}" / "prompt.md"
                ).read_text(encoding="utf-8")
                treated_prompt = (
                    treated.run_dir / "opportunities" / f"{FORK:04d}" / "prompt.md"
                ).read_text(encoding="utf-8")
                opcodes = difflib.SequenceMatcher(
                    a=control_prompt.splitlines(), b=treated_prompt.splitlines()
                ).get_opcodes()
                if any(tag in {"delete", "replace"} for tag, *_ in opcodes):
                    raise ValueError(
                        f"{architecture} B{block} {control_condition}/{treated_condition}: "
                        "fork prompt changes more than an insertion"
                    )
                inserted = [
                    line
                    for tag, _i1, _i2, j1, j2 in opcodes
                    if tag == "insert"
                    for line in treated_prompt.splitlines()[j1:j2]
                ]
                if not inserted or not any("assumption" in line.lower() for line in inserted):
                    raise ValueError(
                        f"{architecture} B{block} {control_condition}/{treated_condition}: "
                        "fork insertion lacks assumption direction"
                    )
                report["fork_prompt_insertion_only"] += 1
    return report


def mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else math.nan


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fork_mechanism_summary(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm, treated in (("ordinary", 0), ("assumption_challenge", 1)):
        arm_rows = [row for row in records if int(row["treated"]) == treated]
        families = sorted({str(row["mechanism_family"]) for row in arm_rows})
        for family in families:
            rows = [row for row in arm_rows if row["mechanism_family"] == family]
            reductions = [safe_float(row["immediate_parameter_reduction"]) for row in rows]
            successful = [value for value in reductions if value > 0]
            output.append(
                {
                    "arm": arm,
                    "mechanism_family": family,
                    "n": len(rows),
                    "share_within_arm": len(rows) / len(arm_rows),
                    "qualified_rate": mean(row["qualified"] for row in rows),
                    "successful_update_rate": mean(value > 0 for value in reductions),
                    "mean_parameter_reduction": mean(reductions),
                    "median_successful_parameter_reduction": (
                        statistics.median(successful) if successful else math.nan
                    ),
                    "mean_structural_novelty": mean(
                        safe_float(row["structural_novelty"]) for row in rows
                    ),
                    "mean_changed_lines": mean(row["changed_lines"] for row in rows),
                }
            )
    return output


def summarize_records(records: Sequence[dict[str, Any]], group_keys: Sequence[str]) -> list[dict[str, Any]]:
    metrics = [
        "structural_novelty",
        "ast_distance",
        "changed_lines",
        "declared_lexical_novelty",
        "new_family_tag",
        "assumption_language",
        "mechanism_shift_language",
        "qualified",
        "retained",
        "became_incumbent",
        "total_tokens",
        "output_tokens",
        "evaluator_seconds",
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in group_keys)].append(record)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        summary = {name: value for name, value in zip(group_keys, key, strict=True)}
        summary["n"] = len(rows)
        for metric in metrics:
            summary[f"{metric}_mean"] = mean(
                safe_float(row[metric]) for row in rows
            )
        summary["qualified_structural_rate"] = mean(
            int(bool(row["qualified"]) and bool(row["structural_edit"])) for row in rows
        )
        summary["retained_structural_rate"] = mean(
            int(bool(row["retained"]) and bool(row["structural_edit"])) for row in rows
        )
        output.append(summary)
    return output


def trajectory_record(run: Run, records: Sequence[dict[str, Any]], horizon: int) -> dict[str, Any]:
    chosen = [
        row
        for row in records
        if row["run_id"] == run.run_id and FORK <= int(row["opportunity"]) <= horizon
    ]
    at_fork = incumbent_parameters_at(run, FORK - 1)
    final_parameters = incumbent_parameters_at(run, horizon)
    transitions = [row for row in chosen if row["opportunity"] % 10 == 0]
    nontransitions = [row for row in chosen if row["opportunity"] % 10 != 0]
    return {
        "architecture": run.architecture,
        "run_id": run.run_id,
        "block": run.block,
        "condition": run.condition,
        "memory": run.memory,
        "treated": int(run.treated),
        "horizon": horizon,
        "fork_parameters": at_fork,
        "final_parameters": final_parameters,
        "parameter_reduction": at_fork - final_parameters,
        "parameter_reduction_fraction": (at_fork - final_parameters) / at_fork,
        "qualified_rate": mean(row["qualified"] for row in chosen),
        "retained_rate": mean(row["retained"] for row in chosen),
        "qualified_structural_rate": mean(
            int(bool(row["qualified"]) and bool(row["structural_edit"])) for row in chosen
        ),
        "structural_novelty_mean": mean(row["structural_novelty"] for row in chosen),
        "declared_lexical_novelty_mean": mean(
            row["declared_lexical_novelty"] for row in chosen
        ),
        "new_family_tag_rate": mean(row["new_family_tag"] for row in chosen),
        "unique_mechanisms": len({row["mechanism"].strip().lower() for row in chosen if row["mechanism"].strip()}),
        "unique_family_signatures": len({row["family_tags"] for row in chosen}),
        "tokens": sum(int(row["total_tokens"]) for row in chosen),
        "output_tokens": sum(int(row["output_tokens"]) for row in chosen),
        "evaluator_seconds": sum(float(row["evaluator_seconds"]) for row in chosen),
        "transition_structural_novelty": mean(row["structural_novelty"] for row in transitions),
        "nontransition_structural_novelty": mean(row["structural_novelty"] for row in nontransitions),
        "transition_qualified_rate": mean(row["qualified"] for row in transitions),
        "nontransition_qualified_rate": mean(row["qualified"] for row in nontransitions),
    }


PAIR_METRICS = [
    "structural_novelty",
    "ast_distance",
    "changed_lines",
    "declared_lexical_novelty",
    "new_family_tag",
    "assumption_language",
    "mechanism_shift_language",
    "qualified",
    "qualified_structural",
    "retained",
    "retained_structural",
    "became_incumbent",
    "immediate_parameter_reduction",
    "total_tokens",
    "output_tokens",
    "evaluator_seconds",
]


def fork_pair_rows(fork_records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {
        (row["architecture"], int(row["block"]), row["condition"]): row
        for row in fork_records
    }
    output: list[dict[str, Any]] = []
    for architecture in ("greedy", "native"):
        for block in range(1, 9):
            for memory, control_condition, treated_condition in (
                ("single", "C0", "C1"),
                ("portfolio", "C2", "C3"),
            ):
                control = mapping[(architecture, block, control_condition)]
                treated = mapping[(architecture, block, treated_condition)]
                row: dict[str, Any] = {
                    "architecture": architecture,
                    "block": block,
                    "memory": memory,
                    "control_condition": control_condition,
                    "treated_condition": treated_condition,
                }
                for metric in PAIR_METRICS:
                    if metric == "qualified_structural":
                        control_value = int(control["qualified"] and control["structural_edit"])
                        treated_value = int(treated["qualified"] and treated["structural_edit"])
                    elif metric == "retained_structural":
                        control_value = int(control["retained"] and control["structural_edit"])
                        treated_value = int(treated["retained"] and treated["structural_edit"])
                    else:
                        control_value = safe_float(control[metric])
                        treated_value = safe_float(treated[metric])
                    row[f"control_{metric}"] = control_value
                    row[f"treated_{metric}"] = treated_value
                    row[f"difference_{metric}"] = treated_value - control_value
                row["control_mechanism"] = control["mechanism"]
                row["treated_mechanism"] = treated["mechanism"]
                output.append(row)
    return output


def paired_bootstrap(
    rows: Sequence[dict[str, Any]], metric: str, repetitions: int = 20000
) -> tuple[float, float, float]:
    clusters: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = safe_float(row[f"difference_{metric}"])
        if math.isfinite(value):
            clusters[(row["architecture"], int(row["block"]))].append(value)
    by_architecture: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in clusters:
        by_architecture[key[0]].append(key)
    rng = random.Random(SEED + sum(ord(char) for char in metric))
    estimates: list[float] = []
    for _ in range(repetitions):
        sampled_values: list[float] = []
        for architecture, keys in sorted(by_architecture.items()):
            del architecture
            for _index in range(len(keys)):
                selected = rng.choice(keys)
                sampled_values.extend(clusters[selected])
        estimates.append(statistics.fmean(sampled_values))
    estimates.sort()
    point = mean(row[f"difference_{metric}"] for row in rows)
    return point, estimates[int(0.025 * repetitions)], estimates[int(0.975 * repetitions)]


def paired_effect_summary(pair_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    subsets: list[tuple[str, list[dict[str, Any]]]] = [("all", list(pair_rows))]
    for architecture in ("greedy", "native"):
        subsets.append((architecture, [row for row in pair_rows if row["architecture"] == architecture]))
    for memory in ("single", "portfolio"):
        subsets.append((memory, [row for row in pair_rows if row["memory"] == memory]))
    for subset_name, rows in subsets:
        for metric in PAIR_METRICS:
            point, low, high = paired_bootstrap(rows, metric, repetitions=10000)
            differences = [safe_float(row[f"difference_{metric}"]) for row in rows]
            output.append(
                {
                    "subset": subset_name,
                    "metric": metric,
                    "n_pairs": len(rows),
                    "control_mean": mean(row[f"control_{metric}"] for row in rows),
                    "treated_mean": mean(row[f"treated_{metric}"] for row in rows),
                    "paired_difference": point,
                    "cluster_bootstrap_low": low,
                    "cluster_bootstrap_high": high,
                    "treated_higher": sum(value > 0 for value in differences),
                    "control_higher": sum(value < 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                }
            )
    return output


def trajectory_pairs(trajectory_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {
        (row["architecture"], int(row["block"]), row["condition"]): row
        for row in trajectory_rows
    }
    metrics = [
        "final_parameters",
        "parameter_reduction",
        "parameter_reduction_fraction",
        "qualified_rate",
        "retained_rate",
        "qualified_structural_rate",
        "structural_novelty_mean",
        "declared_lexical_novelty_mean",
        "new_family_tag_rate",
        "unique_mechanisms",
        "unique_family_signatures",
        "tokens",
        "output_tokens",
        "evaluator_seconds",
    ]
    output: list[dict[str, Any]] = []
    for architecture in ("greedy", "native"):
        for block in range(1, 9):
            for memory, control_condition, treated_condition in (
                ("single", "C0", "C1"),
                ("portfolio", "C2", "C3"),
            ):
                control = mapping[(architecture, block, control_condition)]
                treated = mapping[(architecture, block, treated_condition)]
                row: dict[str, Any] = {
                    "architecture": architecture,
                    "block": block,
                    "memory": memory,
                }
                for metric in metrics:
                    row[f"control_{metric}"] = control[metric]
                    row[f"treated_{metric}"] = treated[metric]
                    row[f"difference_{metric}"] = treated[metric] - control[metric]
                output.append(row)
    return output


def trajectory_effect_summary(pair_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "final_parameters",
        "parameter_reduction",
        "parameter_reduction_fraction",
        "qualified_rate",
        "retained_rate",
        "qualified_structural_rate",
        "structural_novelty_mean",
        "declared_lexical_novelty_mean",
        "new_family_tag_rate",
        "unique_mechanisms",
        "tokens",
        "output_tokens",
        "evaluator_seconds",
    ]
    output: list[dict[str, Any]] = []
    for subset_name, rows in [
        ("all", list(pair_rows)),
        ("greedy", [row for row in pair_rows if row["architecture"] == "greedy"]),
        ("native", [row for row in pair_rows if row["architecture"] == "native"]),
        ("single", [row for row in pair_rows if row["memory"] == "single"]),
        ("portfolio", [row for row in pair_rows if row["memory"] == "portfolio"]),
    ]:
        for metric in metrics:
            point, low, high = paired_bootstrap(rows, metric, repetitions=10000)
            output.append(
                {
                    "subset": subset_name,
                    "metric": metric,
                    "n_pairs": len(rows),
                    "control_mean": mean(row[f"control_{metric}"] for row in rows),
                    "treated_mean": mean(row[f"treated_{metric}"] for row in rows),
                    "paired_difference": point,
                    "cluster_bootstrap_low": low,
                    "cluster_bootstrap_high": high,
                }
            )
    return output


def horizon_effects(runs: Sequence[Run], records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    horizons = [10, 19, 20, 29, 30, 39, 40, 49, 50, 59, 60, 69, 70]
    output: list[dict[str, Any]] = []
    for horizon in horizons:
        summaries = [trajectory_record(run, records, horizon) for run in runs]
        pairs = trajectory_pairs(summaries)
        point, low, high = paired_bootstrap(
            pairs, "final_parameters", repetitions=10000
        )
        differences = [float(row["difference_final_parameters"]) for row in pairs]
        output.append(
            {
                "horizon": horizon,
                "interventions_received": 1 + max(0, (horizon - 10) // 10),
                "n_pairs": len(pairs),
                "control_final_parameters_mean": mean(
                    row["control_final_parameters"] for row in pairs
                ),
                "treated_final_parameters_mean": mean(
                    row["treated_final_parameters"] for row in pairs
                ),
                "paired_difference": point,
                "cluster_bootstrap_low": low,
                "cluster_bootstrap_high": high,
                "treated_lower_count": sum(value < 0 for value in differences),
                "control_lower_count": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
            }
        )
    return output


def fashion_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs = load_runs(FASHION, "fashion_replication")
    if len(runs) != 20 or any(len(run.events) != 200 for run in runs):
        raise ValueError("Fashion-MNIST replication requires 20 complete 200-proposal runs")
    records = [event_record(run, event) for run in runs for event in run.events]
    transitions = [row for row in records if row["opportunity"] % 10 == 0]
    mapping = {
        (row["block"], row["condition"], row["opportunity"]): row
        for row in transitions
    }
    pairs: list[dict[str, Any]] = []
    metrics = [
        "structural_novelty",
        "ast_distance",
        "changed_lines",
        "declared_lexical_novelty",
        "new_family_tag",
        "assumption_language",
        "qualified",
        "retained",
        "became_incumbent",
        "total_tokens",
        "output_tokens",
        "evaluator_seconds",
    ]
    for block in range(1, 6):
        for opportunity in range(10, 201, 10):
            for memory, control_condition, treated_condition in (
                ("single", "C0", "C1"),
                ("portfolio", "C2", "C3"),
            ):
                control = mapping[(block, control_condition, opportunity)]
                treated = mapping[(block, treated_condition, opportunity)]
                row: dict[str, Any] = {
                    "block": block,
                    "scope": "original" if block <= 3 else "extension",
                    "opportunity": opportunity,
                    "memory": memory,
                }
                for metric in metrics:
                    left = safe_float(control[metric])
                    right = safe_float(treated[metric])
                    row[f"control_{metric}"] = left
                    row[f"treated_{metric}"] = right
                    row[f"difference_{metric}"] = right - left
                pairs.append(row)
    return records, pairs


def fashion_summary(records: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "structural_novelty",
        "ast_distance",
        "changed_lines",
        "declared_lexical_novelty",
        "new_family_tag",
        "assumption_language",
        "qualified",
        "retained",
        "became_incumbent",
        "total_tokens",
        "output_tokens",
        "evaluator_seconds",
    ]
    output: list[dict[str, Any]] = []
    for scope in ("all", "original", "extension"):
        selected = list(pairs) if scope == "all" else [row for row in pairs if row["scope"] == scope]
        for metric in metrics:
            differences = [safe_float(row[f"difference_{metric}"]) for row in selected]
            output.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "n_checkpoint_pairs": len(selected),
                    "control_mean": mean(row[f"control_{metric}"] for row in selected),
                    "treated_mean": mean(row[f"treated_{metric}"] for row in selected),
                    "paired_difference": mean(differences),
                    "treated_higher": sum(value > 0 for value in differences),
                    "control_higher": sum(value < 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                }
            )
    # Difference-in-differences at checkpoint versus immediately preceding ordinary proposal.
    mapping = {
        (row["block"], row["condition"], row["opportunity"]): row for row in records
    }
    for scope in ("all", "original", "extension"):
        blocks = range(1, 6) if scope == "all" else (range(1, 4) if scope == "original" else range(4, 6))
        for metric in metrics:
            did_values: list[float] = []
            for block in blocks:
                for opportunity in range(10, 201, 10):
                    for control_condition, treated_condition in (("C0", "C1"), ("C2", "C3")):
                        control_now = safe_float(mapping[(block, control_condition, opportunity)][metric])
                        control_before = safe_float(mapping[(block, control_condition, opportunity - 1)][metric])
                        treated_now = safe_float(mapping[(block, treated_condition, opportunity)][metric])
                        treated_before = safe_float(mapping[(block, treated_condition, opportunity - 1)][metric])
                        if all(math.isfinite(value) for value in (control_now, control_before, treated_now, treated_before)):
                            did_values.append((treated_now - treated_before) - (control_now - control_before))
            output.append(
                {
                    "scope": f"{scope}_checkpoint_did",
                    "metric": metric,
                    "n_checkpoint_pairs": len(did_values),
                    "control_mean": math.nan,
                    "treated_mean": math.nan,
                    "paired_difference": mean(did_values),
                    "treated_higher": sum(value > 0 for value in did_values),
                    "control_higher": sum(value < 0 for value in did_values),
                    "ties": sum(value == 0 for value in did_values),
                }
            )
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty table {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def make_figures(
    records: Sequence[dict[str, Any]],
    fork_effects: Sequence[dict[str, Any]],
    mechanism_summary: Sequence[dict[str, Any]],
    trajectory_rows: Sequence[dict[str, Any]],
    horizon_rows: Sequence[dict[str, Any]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 180,
        }
    )

    selected_metrics = [
        ("structural_novelty", "Normalized source novelty"),
        ("qualified", "Qualification rate"),
        ("retained", "Retention rate"),
        ("output_tokens", "Output tokens"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.65))
    subsets = ["greedy", "native", "single", "portfolio"]
    palette = {"greedy": "#2a6fbb", "native": "#e07a2e", "single": "#6a4c93", "portfolio": "#2a9d8f"}
    effect_map = {(row["subset"], row["metric"]): row for row in fork_effects}
    for axis, (metric, label) in zip(axes, selected_metrics, strict=True):
        for index, subset in enumerate(subsets):
            row = effect_map[(subset, metric)]
            axis.errorbar(
                row["paired_difference"],
                index,
                xerr=[
                    [row["paired_difference"] - row["cluster_bootstrap_low"]],
                    [row["cluster_bootstrap_high"] - row["paired_difference"]],
                ],
                fmt="o",
                color=palette[subset],
                capsize=2.5,
                markersize=4.5,
            )
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_title(label)
        axis.set_yticks(range(len(subsets)), ["Greedy", "Native", "Single", "Portfolio"] if axis is axes[0] else ["", "", "", ""])
        axis.grid(axis="x", color="#dddddd", linewidth=0.6)
        if metric in {"qualified", "retained"}:
            axis.xaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f} pp")
    fig.suptitle("Immediate effect of adding the assumption-challenge direction at the matched fork", y=1.04, fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig1_fork_effects.{suffix}", bbox_inches="tight")
    plt.close(fig)

    run_lookup = {(row["architecture"], row["run_id"]): row for row in trajectory_rows}
    del run_lookup
    conditions = ["C0", "C1", "C2", "C3"]
    colors = {"C0": "#4575b4", "C1": "#d73027", "C2": "#74add1", "C3": "#f46d43"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), sharey=True)
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_run[row["run_id"]].append(row)
    for axis, architecture in zip(axes, ("greedy", "native"), strict=True):
        architecture_runs = [run for run in PRIMARY_RUNS if run.architecture == architecture]
        for condition in conditions:
            curves: list[list[float]] = []
            for run in architecture_runs:
                if run.condition != condition:
                    continue
                curves.append(
                    [incumbent_parameters_at(run, opportunity) for opportunity in range(9, COMMON_HORIZON + 1)]
                )
            array = np.asarray(curves, dtype=float)
            median = np.median(array, axis=0)
            low = np.quantile(array, 0.25, axis=0)
            high = np.quantile(array, 0.75, axis=0)
            x = np.arange(9, COMMON_HORIZON + 1)
            axis.plot(x, median, label=condition, color=colors[condition], linewidth=1.6)
            axis.fill_between(x, low, high, color=colors[condition], alpha=0.12)
        for checkpoint in range(10, COMMON_HORIZON + 1, 10):
            axis.axvline(checkpoint, color="#cccccc", linewidth=0.45, zorder=0)
        axis.set_title(f"{architecture.title()} OpenEvolve")
        axis.set_xlabel("Proposal opportunity")
        axis.grid(axis="y", color="#e2e2e2", linewidth=0.6)
    axes[0].set_ylabel("Incumbent qualified parameters (median, IQR)")
    axes[1].legend(ncol=2, frameon=False, loc="upper right")
    fig.suptitle("Post-fork search trajectories through the common horizon", y=1.02, fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig2_parameter_trajectories.{suffix}", bbox_inches="tight")
    plt.close(fig)

    fork_rows = [row for row in records if row["opportunity"] == FORK]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.15))
    jitter_rng = random.Random(SEED)
    for axis, architecture in zip(axes, ("greedy", "native"), strict=True):
        rows = [row for row in fork_rows if row["architecture"] == architecture]
        for treated, marker, label, color in (
            (0, "o", "Ordinary", "#4575b4"),
            (1, "^", "Assumption challenge", "#d73027"),
        ):
            chosen = [row for row in rows if row["treated"] == treated]
            axis.scatter(
                [row["structural_novelty"] for row in chosen],
                [row["parameters"] if row["parameters"] is not None else 26000 + jitter_rng.random() * 500 for row in chosen],
                marker=marker,
                color=color,
                alpha=0.75,
                s=35,
                label=label,
                edgecolor="white",
                linewidth=0.4,
            )
        axis.axhline(25000, color="#555555", linestyle="--", linewidth=0.7)
        axis.set_title(architecture.title())
        axis.set_xlabel("Normalized source novelty")
        axis.grid(color="#e2e2e2", linewidth=0.6)
    axes[0].set_ylabel("Candidate parameters (unqualified above dashed line if unavailable)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Novelty-feasibility trade-off at the matched fork", y=1.02, fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig3_novelty_feasibility.{suffix}", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 3.0))
    x = [int(row["horizon"]) for row in horizon_rows]
    y = [float(row["paired_difference"]) for row in horizon_rows]
    low = [float(row["cluster_bootstrap_low"]) for row in horizon_rows]
    high = [float(row["cluster_bootstrap_high"]) for row in horizon_rows]
    axis.plot(x, y, marker="o", color="#7b2cbf", linewidth=1.7, markersize=3.8)
    axis.fill_between(x, low, high, color="#7b2cbf", alpha=0.16)
    axis.axhline(0, color="#333333", linewidth=0.8)
    for checkpoint in range(10, COMMON_HORIZON + 1, 10):
        axis.axvline(checkpoint, color="#d8d8d8", linewidth=0.5, zorder=0)
    axis.set_xlabel("Common-horizon proposal")
    axis.set_ylabel("Treated - control incumbent parameters")
    axis.set_title("The compression advantage emerges after the matched fork")
    axis.grid(axis="y", color="#e2e2e2", linewidth=0.6)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig4_horizon_effect.{suffix}", bbox_inches="tight")
    plt.close(fig)

    family_order = [
        "normalization_or_bias_pruning",
        "feedforward_compression",
        "relative_position_attention",
        "projection_reuse",
        "iterative_or_shared_depth",
        "attention_routing_reparameterization",
        "token_interface_factorization",
    ]
    family_labels = [
        "Normalization / bias pruning",
        "Feedforward compression",
        "Relative-position attention",
        "Cross-sublayer projection reuse",
        "Iterative / shared depth",
        "Attention-routing reparameterization",
        "Token-interface factorization",
    ]
    mechanism_map = {
        (row["arm"], row["mechanism_family"]): int(row["n"])
        for row in mechanism_summary
    }
    y = np.arange(len(family_order))
    ordinary = [mechanism_map.get(("ordinary", family), 0) for family in family_order]
    challenged = [
        mechanism_map.get(("assumption_challenge", family), 0)
        for family in family_order
    ]
    fig, axis = plt.subplots(figsize=(7.3, 3.25))
    axis.barh(y - 0.18, ordinary, height=0.34, color="#4575b4", label="Ordinary")
    axis.barh(
        y + 0.18,
        challenged,
        height=0.34,
        color="#d73027",
        label="Assumption challenge",
    )
    for index, value in enumerate(ordinary):
        if value:
            axis.text(value + 0.25, index - 0.18, str(value), va="center", fontsize=8)
    for index, value in enumerate(challenged):
        if value:
            axis.text(value + 0.25, index + 0.18, str(value), va="center", fontsize=8)
    axis.set_yticks(y, family_labels)
    axis.set_xlim(0, 20)
    axis.set_xlabel("Number of matched-fork proposals (32 per arm)")
    axis.set_title("The prompt redirects local pruning toward alternative computational mechanisms")
    axis.grid(axis="x", color="#e2e2e2", linewidth=0.6)
    axis.legend(frameon=False, loc="center right")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"fig5_mechanism_taxonomy.{suffix}", bbox_inches="tight")
    plt.close(fig)


PRIMARY_RUNS: list[Run] = []


def analyse(output: Path) -> dict[str, Any]:
    global PRIMARY_RUNS  # Used only by deterministic figure assembly.
    PRIMARY_RUNS = load_runs(GREEDY, "greedy") + load_runs(NATIVE, "native")
    integrity = validate_primary(PRIMARY_RUNS)
    records = [
        event_record(run, event)
        for run in PRIMARY_RUNS
        for event in run.events
        if int(event["opportunity"]) <= COMMON_HORIZON
    ]
    if len(records) != 64 * COMMON_HORIZON:
        raise ValueError(f"Expected 4,480 common-horizon rows, found {len(records)}")
    fork_records = [row for row in records if row["opportunity"] == FORK]
    pairs = fork_pair_rows(fork_records)
    fork_effects = paired_effect_summary(pairs)
    mechanism_summary = fork_mechanism_summary(fork_records)
    trajectory_rows = [
        trajectory_record(run, records, COMMON_HORIZON) for run in PRIMARY_RUNS
    ]
    trajectory_pair_rows = trajectory_pairs(trajectory_rows)
    trajectory_effects = trajectory_effect_summary(trajectory_pair_rows)
    phase1_rows = [trajectory_record(run, records, 19) for run in PRIMARY_RUNS]
    phase1_pair_rows = trajectory_pairs(phase1_rows)
    phase1_effects = trajectory_effect_summary(phase1_pair_rows)
    horizon_rows = horizon_effects(PRIMARY_RUNS, records)
    fashion_rows, fashion_pairs = fashion_records()
    fashion_effects = fashion_summary(fashion_rows, fashion_pairs)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "primary_proposals.csv", records)
    write_csv(output / "fork_proposals.csv", fork_records)
    write_csv(output / "fork_pairs.csv", pairs)
    write_csv(output / "fork_effects.csv", fork_effects)
    write_csv(output / "fork_mechanism_summary.csv", mechanism_summary)
    write_csv(output / "trajectory_summary.csv", trajectory_rows)
    write_csv(output / "trajectory_pairs.csv", trajectory_pair_rows)
    write_csv(output / "trajectory_effects.csv", trajectory_effects)
    write_csv(output / "phase1_trajectory_summary.csv", phase1_rows)
    write_csv(output / "phase1_pairs.csv", phase1_pair_rows)
    write_csv(output / "phase1_effects.csv", phase1_effects)
    write_csv(output / "horizon_effects.csv", horizon_rows)
    write_csv(output / "fashion_proposals.csv", fashion_rows)
    write_csv(output / "fashion_checkpoint_pairs.csv", fashion_pairs)
    write_csv(output / "fashion_effects.csv", fashion_effects)

    make_figures(
        records,
        fork_effects,
        mechanism_summary,
        trajectory_rows,
        horizon_rows,
        output,
    )
    aggregate = {
        "analysis_seed": SEED,
        "primary_campaigns": [
            str(GREEDY.relative_to(REPO)),
            str(NATIVE.relative_to(REPO)),
        ],
        "primary_integrity": integrity,
        "primary_runs": len(PRIMARY_RUNS),
        "primary_common_horizon": COMMON_HORIZON,
        "primary_rows": len(records),
        "primary_fork_pairs": len(pairs),
        "primary_recorded_messages": sum(
            int(
                (
                    run.run_dir
                    / "opportunities"
                    / f"{opportunity:04d}"
                    / "codex"
                    / f"proposal-{opportunity}.last-message.md"
                ).exists()
            )
            for run in PRIMARY_RUNS
            for opportunity in range(1, COMMON_HORIZON + 1)
        ),
        "fashion_campaign": str(FASHION.relative_to(REPO)),
        "fashion_runs": 20,
        "fashion_rows": len(fashion_rows),
        "fashion_checkpoint_pairs": len(fashion_pairs),
        "fork_all_effects": {
            row["metric"]: {
                "control_mean": row["control_mean"],
                "treated_mean": row["treated_mean"],
                "paired_difference": row["paired_difference"],
                "cluster_bootstrap_low": row["cluster_bootstrap_low"],
                "cluster_bootstrap_high": row["cluster_bootstrap_high"],
            }
            for row in fork_effects
            if row["subset"] == "all"
        },
        "fork_success_magnitude": {
            label: {
                "successful_updates": len(values),
                "conditional_mean_parameter_reduction": mean(values),
                "conditional_median_parameter_reduction": statistics.median(values),
            }
            for label, values in {
                "control": [
                    float(row["control_immediate_parameter_reduction"])
                    for row in pairs
                    if float(row["control_immediate_parameter_reduction"]) > 0
                ],
                "treated": [
                    float(row["treated_immediate_parameter_reduction"])
                    for row in pairs
                    if float(row["treated_immediate_parameter_reduction"]) > 0
                ],
            }.items()
        },
        "fork_mechanism_counts": {
            arm: {
                row["mechanism_family"]: row["n"]
                for row in mechanism_summary
                if row["arm"] == arm
            }
            for arm in ("ordinary", "assumption_challenge")
        },
        "horizon70_all_effects": {
            row["metric"]: {
                "control_mean": row["control_mean"],
                "treated_mean": row["treated_mean"],
                "paired_difference": row["paired_difference"],
                "cluster_bootstrap_low": row["cluster_bootstrap_low"],
                "cluster_bootstrap_high": row["cluster_bootstrap_high"],
            }
            for row in trajectory_effects
            if row["subset"] == "all"
        },
        "phase1_all_effects": {
            row["metric"]: {
                "control_mean": row["control_mean"],
                "treated_mean": row["treated_mean"],
                "paired_difference": row["paired_difference"],
                "cluster_bootstrap_low": row["cluster_bootstrap_low"],
                "cluster_bootstrap_high": row["cluster_bootstrap_high"],
            }
            for row in phase1_effects
            if row["subset"] == "all"
        },
    }
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    aggregate = analyse(args.output.resolve())
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
