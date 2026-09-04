#!/usr/bin/env python3
"""Generate Paper 6.1 robustness and descendant-branch summaries.

This script operates only on the deterministic proposal and matched-pair tables
written by ``analysis.py``.  It adds block-level sign checks, leave-one-block-
out estimates, and descriptive descendant outcomes conditional on whether the
scheduled challenged proposal was retained.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"

TASKS = ("addition", "fashion", "nanogpt")
METRICS = (
    "lexical_novelty",
    "source_novelty",
    "ast_distance",
    "changed_lines",
    "valid",
    "retained",
    "output_tokens",
)

WORD_RE = re.compile(r"[a-z][a-z0-9_+-]{1,}")
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


def read_csv(name: str) -> list[dict[str, str]]:
    with (DERIVED / name).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite(values: Iterable[Any]) -> list[float]:
    output = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def mean(values: Iterable[Any]) -> float:
    clean = finite(values)
    return statistics.fmean(clean) if clean else math.nan


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    path = DERIVED / name
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    """Two-sided exact sign-test p-value after discarding zero differences."""
    n = positive + negative
    if n == 0:
        return math.nan
    extreme = max(positive, negative)
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1)) / (2**n)
    return min(1.0, 2 * tail)


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def mechanism_tags(text: str) -> set[str]:
    return {name for name, pattern in FAMILY_PATTERNS.items() if pattern.search(text or "")}


def primary_family(text: str) -> str:
    matches = []
    for name, pattern in FAMILY_PATTERNS.items():
        match = pattern.search(text or "")
        if match:
            matches.append((match.start(), name))
    return min(matches)[1] if matches else "unclassified"


def mechanism_words(text: str) -> set[str]:
    return set(WORD_RE.findall((text or "").lower()))


def pairwise_distance(values: list[set[str]]) -> float:
    distances = [jaccard(left, right) for index, left in enumerate(values) for right in values[index + 1 :]]
    return mean(distances)


def population_sensitivity(proposals: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoints = [row for row in proposals if int(row["checkpoint"])]
    by_checkpoint = []
    for task in TASKS:
        opportunities = sorted({int(row["opportunity"]) for row in checkpoints if row["task"] == task})
        for opportunity in opportunities:
            for arm, treated in (("ordinary", 0), ("assumption_challenge", 1)):
                rows = [
                    row for row in checkpoints
                    if row["task"] == task and int(row["opportunity"]) == opportunity and int(row["treated"]) == treated
                ]
                tag_sets = [mechanism_tags(row["mechanism"]) for row in rows]
                primary_sets = [{primary_family(row["mechanism"])} for row in rows]
                lexical_sets = [mechanism_words(row["mechanism"]) for row in rows]
                by_checkpoint.append(
                    {
                        "task": task,
                        "opportunity": opportunity,
                        "arm": arm,
                        "n_runs": len(rows),
                        "mean_mechanism_tag_count": mean(len(value) for value in tag_sets),
                        "mean_mechanism_words": mean(len(value) for value in lexical_sets),
                        "mechanism_only_family_distance": pairwise_distance(tag_sets),
                        "primary_family_distance": pairwise_distance(primary_sets),
                        "mechanism_lexical_distance": pairwise_distance(lexical_sets),
                    }
                )

    full = read_csv("population_dispersion.csv")
    summary = []
    for task in TASKS:
        for arm in ("ordinary", "assumption_challenge"):
            selected = [row for row in by_checkpoint if row["task"] == task and row["arm"] == arm]
            full_selected = [row for row in full if row["task"] == task and row["arm"] == arm]
            summary.append(
                {
                    "task": task,
                    "arm": arm,
                    "n_checkpoints": len(selected),
                    "mean_mechanism_tag_count": mean(row["mean_mechanism_tag_count"] for row in selected),
                    "mean_mechanism_words": mean(row["mean_mechanism_words"] for row in selected),
                    "mechanism_only_family_distance": mean(row["mechanism_only_family_distance"] for row in selected),
                    "primary_family_distance": mean(row["primary_family_distance"] for row in selected),
                    "mechanism_lexical_distance": mean(row["mechanism_lexical_distance"] for row in selected),
                    "full_rationale_family_distance": mean(row["between_run_family_distance"] for row in full_selected),
                }
            )
    return by_checkpoint, summary


def lineage_descendants(proposals: list[dict[str, str]], checkpoints: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_run_opportunity = {
        (row["task"], row["run_id"], int(row["opportunity"])): row
        for row in proposals
    }
    parents: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in proposals:
        if row["candidate_id"]:
            parents[(row["task"], row["run_id"])][row["candidate_id"]] = row["parent_id"]

    def descends(task: str, run_id: str, candidate_id: str, anchor: str) -> bool:
        seen: set[str] = set()
        current = candidate_id
        mapping = parents[(task, run_id)]
        while current and current not in seen:
            seen.add(current)
            parent = mapping.get(current, "")
            if parent == anchor:
                return True
            current = parent
        return False

    rows = []
    for pair in checkpoints:
        task = pair["task"]
        checkpoint = int(pair["opportunity"])
        horizon = {"addition": 80, "fashion": 200, "nanogpt": 40}[task]
        end = min(checkpoint + 9, horizon)
        output: dict[str, Any] = {
            "task": task,
            "block": int(pair["block"]),
            "memory": pair["memory"],
            "checkpoint": checkpoint,
            "cycle_end": end,
        }
        for arm in ("control", "treated"):
            run_id = pair[f"{arm}_run_id"]
            anchor_row = by_run_opportunity[(task, run_id, checkpoint)]
            anchor = anchor_row["candidate_id"]
            followups = [by_run_opportunity[(task, run_id, opportunity)] for opportunity in range(checkpoint + 1, end + 1)]
            descendants = [row for row in followups if anchor and descends(task, run_id, row["candidate_id"], anchor)]
            retained_descendants = [row for row in descendants if int(row["retained"])]
            output[f"{arm}_anchor_retained"] = int(anchor_row["retained"])
            output[f"{arm}_descendant_proposals"] = len(descendants)
            output[f"{arm}_retained_descendants"] = len(retained_descendants)
            output[f"{arm}_descendant_gain"] = sum(float(row["incumbent_gain"]) for row in retained_descendants)
            output[f"{arm}_branch_gain"] = (float(anchor_row["incumbent_gain"]) if int(anchor_row["retained"]) else 0.0) + output[f"{arm}_descendant_gain"]
        for metric in ("anchor_retained", "descendant_proposals", "retained_descendants", "descendant_gain", "branch_gain"):
            output[f"difference_{metric}"] = output[f"treated_{metric}"] - output[f"control_{metric}"]
        rows.append(output)

    summary = []
    for task in TASKS:
        for memory in ("all", "single-incumbent", "four-lineage"):
            selected = [row for row in rows if row["task"] == task and (memory == "all" or row["memory"] == memory)]
            record: dict[str, Any] = {"task": task, "memory": memory, "n_cycles": len(selected)}
            for arm in ("control", "treated"):
                record[f"{arm}_anchor_retention_rate"] = mean(row[f"{arm}_anchor_retained"] for row in selected)
                record[f"{arm}_cycles_with_descendants"] = sum(row[f"{arm}_descendant_proposals"] > 0 for row in selected)
                record[f"{arm}_mean_descendant_proposals"] = mean(row[f"{arm}_descendant_proposals"] for row in selected)
                record[f"{arm}_mean_retained_descendants"] = mean(row[f"{arm}_retained_descendants"] for row in selected)
                record[f"{arm}_mean_descendant_gain"] = mean(row[f"{arm}_descendant_gain"] for row in selected)
                record[f"{arm}_mean_branch_gain"] = mean(row[f"{arm}_branch_gain"] for row in selected)
            for metric in ("descendant_proposals", "retained_descendants", "descendant_gain", "branch_gain"):
                record[f"paired_mean_difference_{metric}"] = mean(row[f"difference_{metric}"] for row in selected)
            summary.append(record)
    return rows, summary


def source_missingness(proposals: list[dict[str, str]], checkpoints: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    checkpoint_rows = [row for row in proposals if int(row["checkpoint"])]
    for task in TASKS:
        ordinary = [row for row in checkpoint_rows if row["task"] == task and not int(row["treated"])]
        challenged = [row for row in checkpoint_rows if row["task"] == task and int(row["treated"])]
        task_pairs = [row for row in checkpoints if row["task"] == task]
        output.append(
            {
                "task": task,
                "checkpoint_pairs": len(task_pairs),
                "ordinary_source_available": sum(int(row["source_available"]) for row in ordinary),
                "ordinary_checkpoint_proposals": len(ordinary),
                "ordinary_source_available_rate": mean(row["source_available"] for row in ordinary),
                "challenged_source_available": sum(int(row["source_available"]) for row in challenged),
                "challenged_checkpoint_proposals": len(challenged),
                "challenged_source_available_rate": mean(row["source_available"] for row in challenged),
                "finite_local_did_pairs": sum(math.isfinite(float(row["did_source_novelty"])) for row in task_pairs),
            }
        )
    return output


def block_and_lobo(checkpoints: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    block_rows: list[dict[str, Any]] = []
    lobo_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for metric in METRICS:
        pooled_block_values = []
        for task in TASKS:
            task_rows = [row for row in checkpoints if row["task"] == task]
            blocks = sorted({int(row["block"]) for row in task_rows})
            by_block: dict[int, float] = {}
            for block in blocks:
                value = mean(
                    row[f"did_{metric}"]
                    for row in task_rows
                    if int(row["block"]) == block
                )
                by_block[block] = value
                pooled_block_values.append(value)
                block_rows.append(
                    {
                        "task": task,
                        "block": block,
                        "metric": metric,
                        "block_mean_local_did": value,
                        "sign": 1 if value > 0 else -1 if value < 0 else 0,
                    }
                )

            for omitted in blocks:
                retained = [value for block, value in by_block.items() if block != omitted]
                lobo_rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "omitted_block": omitted,
                        "leave_one_block_out_mean": mean(retained),
                    }
                )

        positive = sum(value > 0 for value in pooled_block_values)
        negative = sum(value < 0 for value in pooled_block_values)
        zero = sum(value == 0 for value in pooled_block_values)
        summary[metric] = {
            "positive_blocks": positive,
            "negative_blocks": negative,
            "zero_blocks": zero,
            "exact_two_sided_sign_p": exact_two_sided_sign_p(positive, negative),
        }

    return block_rows, lobo_rows, summary


def descendant_summary(
    proposals: list[dict[str, str]],
    checkpoints: list[dict[str, str]],
    cycles: list[dict[str, str]],
) -> list[dict[str, Any]]:
    retained = {
        (row["task"], row["run_id"], int(row["opportunity"])): int(row["retained"])
        for row in proposals
    }
    treated_runs = {
        (row["task"], int(row["block"]), row["memory"], int(row["opportunity"])): row["treated_run_id"]
        for row in checkpoints
    }
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in cycles:
        key = (row["task"], int(row["block"]), row["memory"], int(row["checkpoint"]))
        run_id = treated_runs[key]
        was_retained = retained[(row["task"], run_id, int(row["checkpoint"]))]
        grouped[(row["task"], was_retained)].append(row)

    output = []
    for task in TASKS:
        for was_retained in (0, 1):
            rows = grouped[(task, was_retained)]
            output.append(
                {
                    "task": task,
                    "challenge_retained": was_retained,
                    "n_cycles": len(rows),
                    "mean_treated_followup_gain": mean(row["treated_followup_gain"] for row in rows),
                    "median_treated_followup_gain": statistics.median(finite(row["treated_followup_gain"] for row in rows)),
                    "positive_treated_followup_cycles": sum(float(row["treated_followup_gain"]) > 0 for row in rows),
                    "mean_matched_cycle_gain_difference": mean(row["difference_cycle_gain"] for row in rows),
                }
            )
    return output


def main() -> None:
    checkpoints = read_csv("checkpoint_pairs.csv")
    proposals = read_csv("proposal_records.csv")
    cycles = read_csv("cycle_gain_pairs.csv")
    block_rows, lobo_rows, sign_summary = block_and_lobo(checkpoints)
    descendants = descendant_summary(proposals, checkpoints, cycles)
    population_checkpoints, population_summary = population_sensitivity(proposals)
    lineage_rows, lineage_summary = lineage_descendants(proposals, checkpoints)
    missingness = source_missingness(proposals, checkpoints)
    write_csv("block_checkpoint_effects.csv", block_rows)
    write_csv("leave_one_block_out.csv", lobo_rows)
    write_csv("descendant_branch_summary.csv", descendants)
    write_csv("population_measure_sensitivity_checkpoints.csv", population_checkpoints)
    write_csv("population_measure_sensitivity.csv", population_summary)
    write_csv("lineage_descendant_cycles.csv", lineage_rows)
    write_csv("lineage_descendant_summary.csv", lineage_summary)
    write_csv("source_missingness.csv", missingness)
    payload = {
        "block_sign_summary": sign_summary,
        "descendant_branch_summary": descendants,
        "population_measure_sensitivity": population_summary,
        "lineage_descendant_summary": lineage_summary,
        "source_missingness": missingness,
    }
    (DERIVED / "robustness_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
