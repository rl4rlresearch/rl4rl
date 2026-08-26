"""Condition-common multi-fidelity evaluation and developmental credit."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .spec import ObjectiveDirection, TaskSpec
from .state import Evaluation, append_jsonl, atomic_json, utc_now


def _literal_assignment(path: Path, symbol: str) -> object | None:
    """Read one literal module assignment without importing candidate code."""

    if not path.is_file():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None
    found: object | None = None
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is None or not any(
            isinstance(target, ast.Name) and target.id == symbol for target in targets
        ):
            continue
        try:
            found = ast.literal_eval(value)
        except (ValueError, TypeError):
            return None
    return found


def _candidate_ladder_policy(
    *,
    candidate_snapshot: Path,
    config: dict[str, Any],
    default_levels: list[int],
    default_thresholds: list[float | None],
) -> tuple[list[int], list[float | None], dict[str, Any]]:
    """Resolve an optional candidate-authored ladder under evaluator-owned bounds.

    Candidates may choose intermediate evaluation rungs and, for screening
    ladders, promotion thresholds. They cannot raise the common ceiling,
    remove the mandatory terminal rung, exceed the rung-count cap, or bypass
    full-fidelity confirmation.
    """

    policy = dict(config.get("candidate_editable_policy", {}))
    receipt: dict[str, Any] = {
        "enabled": bool(policy.get("enabled", False)),
        "source": "controller_default",
        "accepted": False,
        "reason": "candidate-editable policy disabled",
    }
    if not receipt["enabled"]:
        return default_levels, default_thresholds, receipt
    relative = Path(str(policy.get("path", "")))
    root = candidate_snapshot.resolve()
    source = (root / relative).resolve()
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or root not in source.parents
    ):
        raise ValueError("candidate ladder policy path is unsafe")
    levels_symbol = str(policy.get("levels_symbol", "EVALUATION_LADDER"))
    raw_levels = _literal_assignment(source, levels_symbol)
    if raw_levels is None:
        receipt["reason"] = f"{levels_symbol} literal was not present"
        return default_levels, default_thresholds, receipt
    if not isinstance(raw_levels, list | tuple):
        receipt["reason"] = f"{levels_symbol} was not an integer sequence"
        return default_levels, default_thresholds, receipt
    try:
        levels = [int(value) for value in raw_levels]
    except (TypeError, ValueError):
        receipt["reason"] = f"{levels_symbol} was not an integer sequence"
        return default_levels, default_thresholds, receipt
    minimum = int(policy.get("minimum_level", default_levels[0]))
    maximum = int(policy.get("maximum_level", default_levels[-1]))
    max_rungs = int(policy.get("max_rungs", len(default_levels)))
    terminal = int(policy.get("required_terminal_level", default_levels[-1]))
    if terminal != maximum:
        raise ValueError("required ladder terminal must equal the common maximum")
    if terminal not in levels:
        levels.append(terminal)
    levels = sorted(set(levels))
    if (
        not levels
        or any(level < minimum or level > maximum for level in levels)
        or len(levels) > max_rungs
    ):
        receipt["reason"] = "candidate levels exceeded evaluator-owned bounds"
        return default_levels, default_thresholds, receipt
    thresholds = default_thresholds
    thresholds_symbol = str(
        policy.get("thresholds_symbol", "EVALUATION_PROMOTION_THRESHOLDS")
    )
    if default_thresholds:
        raw_thresholds = _literal_assignment(source, thresholds_symbol)
        if raw_thresholds is None:
            receipt["reason"] = (
                f"custom levels require literal {thresholds_symbol} thresholds"
            )
            return default_levels, default_thresholds, receipt
        if not isinstance(raw_thresholds, list | tuple):
            receipt["reason"] = f"{thresholds_symbol} was not a numeric sequence"
            return default_levels, default_thresholds, receipt
        try:
            thresholds = [
                None if value is None else float(value) for value in raw_thresholds
            ]
        except (TypeError, ValueError):
            receipt["reason"] = f"{thresholds_symbol} was not a numeric sequence"
            return default_levels, default_thresholds, receipt
        if len(thresholds) == len(levels) - 1:
            thresholds.append(None)
        if len(thresholds) != len(levels):
            receipt["reason"] = "candidate levels and thresholds have different sizes"
            return default_levels, default_thresholds, receipt
        if any(
            threshold is not None and not 0.0 <= threshold <= 1.0
            for threshold in thresholds
        ):
            receipt["reason"] = "candidate promotion thresholds must be in [0, 1]"
            return default_levels, default_thresholds, receipt
        thresholds[-1] = None
    receipt.update(
        {
            "source": relative.as_posix(),
            "accepted": True,
            "reason": "safe literal candidate policy accepted",
            "levels_symbol": levels_symbol,
            "thresholds_symbol": thresholds_symbol if default_thresholds else None,
            "levels": levels,
            "promotion_thresholds": thresholds,
            "enforced_minimum_level": minimum,
            "enforced_maximum_level": maximum,
            "enforced_max_rungs": max_rungs,
        }
    )
    return levels, thresholds, receipt


def _replace_command_argument(
    command: tuple[str, ...], *, flag: str, value: int
) -> tuple[str, ...]:
    items = list(command)
    if flag not in items:
        raise ValueError(f"evaluation command has no configurable {flag} argument")
    index = items.index(flag)
    if index + 1 >= len(items):
        raise ValueError(f"evaluation command ends after {flag}")
    items[index + 1] = str(value)
    return tuple(items)


def evaluate_training_ladder(
    *,
    task: TaskSpec,
    config: dict[str, Any],
    candidate_snapshot: Path,
    opportunity_root: Path,
    timeout_seconds: int,
    run_seed: int,
    evaluator_factory: Callable[[TaskSpec], Any],
) -> Evaluation:
    """Run one evaluator workflow whose stages are charged exactly once.

    ``successive_screen_then_full_confirmation_v1`` stops weak candidates early
    and never retains a candidate without the final stage.  The separate
    ``escalate_until_qualified_v1`` mode supports tasks such as AdderBoard where
    training steps are not part of the objective: it increases the training
    budget uniformly until qualification or the configured ceiling.
    """

    strategy = str(config.get("strategy", ""))
    default_levels = [
        int(value)
        for value in config.get("levels", config.get("training_examples", []))
    ]
    flag = str(
        config.get(
            "command_argument",
            "--training-examples" if "training_examples" in config else "--max-steps",
        )
    )
    if (
        not default_levels
        or any(value < 1 for value in default_levels)
        or default_levels != sorted(set(default_levels))
    ):
        raise ValueError("training ladder levels must be positive, sorted, and unique")
    default_thresholds = list(config.get("promotion_validation_accuracy", []))
    if strategy == "successive_screen_then_full_confirmation_v1" and len(
        default_thresholds
    ) != len(default_levels):
        raise ValueError("screening ladder requires one promotion threshold per level")
    levels, thresholds, policy_receipt = _candidate_ladder_policy(
        candidate_snapshot=candidate_snapshot,
        config=config,
        default_levels=default_levels,
        default_thresholds=default_thresholds,
    )
    stages = []
    total_seconds = 0.0
    final: Evaluation | None = None
    for index, level in enumerate(levels):
        stage_task = replace(
            task,
            evaluator_command=_replace_command_argument(
                task.evaluator_command, flag=flag, value=level
            ),
        )
        artifacts = evaluator_factory(stage_task).evaluate(
            candidate_snapshot=candidate_snapshot,
            opportunity_root=opportunity_root
            / "fidelity"
            / f"stage-{index + 1:02d}-{level}",
            timeout_seconds=timeout_seconds,
            run_seed=run_seed,
        )
        evaluation = artifacts.evaluation
        total_seconds += evaluation.evaluator_seconds
        stages.append(
            {
                "stage": index + 1,
                "level": level,
                "valid": evaluation.valid,
                "fitness": evaluation.fitness,
                "metrics": evaluation.metrics,
                "failure_kind": evaluation.failure_kind,
                "evaluator_seconds": evaluation.evaluator_seconds,
            }
        )
        if not evaluation.valid:
            final = Evaluation(
                valid=False,
                fitness=None,
                metrics={
                    "fidelity_highest_level": level,
                    "fidelity_reached_full": False,
                    "fidelity_stages": stages,
                },
                evaluator_seconds=total_seconds,
                evaluator_calls=1,
                failure_kind=evaluation.failure_kind or "fidelity_stage_failure",
            )
            break
        if strategy == "escalate_until_qualified_v1":
            metric = task.qualification_metric
            minimum = task.qualification_minimum
            qualified = metric is None or (
                isinstance(evaluation.metrics.get(metric), int | float)
                and float(evaluation.metrics[metric]) >= float(minimum)
            )
            if qualified:
                final = Evaluation(
                    valid=True,
                    fitness=evaluation.fitness,
                    metrics=evaluation.metrics
                    | {
                        "fidelity_highest_level": level,
                        "fidelity_reached_full": level == levels[-1],
                        "fidelity_qualification_level": level,
                        "fidelity_stages": stages,
                    },
                    evaluator_seconds=total_seconds,
                    evaluator_calls=1,
                )
                break
        elif strategy == "successive_screen_then_full_confirmation_v1":
            if index == len(levels) - 1:
                final = Evaluation(
                    valid=True,
                    fitness=evaluation.fitness,
                    metrics=evaluation.metrics
                    | {
                        "fidelity_highest_level": level,
                        "fidelity_reached_full": True,
                        "fidelity_stages": stages,
                    },
                    evaluator_seconds=total_seconds,
                    evaluator_calls=1,
                )
                break
            threshold = thresholds[index]
            accuracy = evaluation.metrics.get("validation_accuracy")
            if not isinstance(accuracy, int | float) or float(accuracy) < float(
                threshold
            ):
                final = Evaluation(
                    valid=False,
                    fitness=None,
                    metrics={
                        "fidelity_highest_level": level,
                        "fidelity_screen_accuracy": accuracy,
                        "fidelity_promotion_threshold": threshold,
                        "fidelity_reached_full": False,
                        "fidelity_stages": stages,
                    },
                    evaluator_seconds=total_seconds,
                    evaluator_calls=1,
                    failure_kind="fidelity_screen_not_promoted",
                )
                break
        else:
            raise ValueError(f"unknown training ladder strategy: {strategy}")
    if final is None:
        last = stages[-1]
        final = Evaluation(
            valid=False,
            fitness=None,
            metrics={
                "fidelity_highest_level": last["level"],
                "fidelity_reached_full": last["level"] == levels[-1],
                "fidelity_stages": stages,
            },
            evaluator_seconds=total_seconds,
            evaluator_calls=1,
            failure_kind="training_ladder_exhausted_without_qualification",
        )
    atomic_json(
        opportunity_root / "fidelity" / "ladder-result.json",
        {
            "schema_version": "1.0",
            "strategy": strategy,
            "command_argument": flag,
            "levels": levels,
            "candidate_editable_policy": policy_receipt,
            "evaluation": asdict(final),
            "stages": stages,
        },
    )
    return final


def assess_developmental_value(
    *,
    run_dir: Path,
    task: TaskSpec,
    record: dict[str, Any],
    provenance: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Assign non-selective credit and maintain a bounded provisional archive."""

    if not bool(config.get("enabled", False)):
        return {}
    evaluation = dict(record.get("evaluation", {}))
    valid = bool(evaluation.get("valid"))
    retained = bool(record.get("retained"))
    failure_kind = evaluation.get("failure_kind")
    metrics = (
        evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {}
    )
    archive_path = run_dir / "developmental-archive.json"
    archive = (
        json.loads(archive_path.read_text(encoding="utf-8"))
        if archive_path.is_file()
        else {"schema_version": "1.0", "items": []}
    )
    items = list(archive.get("items", []))
    prior_fingerprints = {str(item.get("semantic_delta_fingerprint")) for item in items}
    prior_mechanisms = {
        str(item.get("mechanism", "")).strip().casefold() for item in items
    }
    fingerprint = str((provenance or {}).get("semantic_delta_fingerprint", ""))
    mechanism = str(record.get("mechanism", "")).strip()
    novel = bool(fingerprint) and fingerprint not in prior_fingerprints
    if mechanism and mechanism.casefold() not in {
        "[not recorded]",
        "[missing mechanism]",
    }:
        novel = novel and mechanism.casefold() not in prior_mechanisms
    parent_id = str(record.get("parent_ids", [""])[0])
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    parent = state.get("candidates", {}).get(parent_id, {})
    parent_fitness = parent.get("fitness")
    candidate_fitness = evaluation.get("fitness")
    near = False
    relative_gap = None
    if isinstance(parent_fitness, int | float) and isinstance(
        candidate_fitness, int | float
    ):
        scale = max(abs(float(parent_fitness)), 1.0)
        if task.objective_direction is ObjectiveDirection.MAXIMIZE:
            relative_gap = max(
                0.0, (float(parent_fitness) - float(candidate_fitness)) / scale
            )
        else:
            relative_gap = max(
                0.0, (float(candidate_fitness) - float(parent_fitness)) / scale
            )
        near = relative_gap <= float(config.get("near_incumbent_relative_margin", 0.02))
    components = {
        "valid_execution": float(config.get("valid_execution_credit", 0.25))
        if valid
        else 0.0,
        "novel_delta": float(config.get("novel_delta_credit", 0.25)) if novel else 0.0,
        "near_incumbent": float(config.get("near_incumbent_credit", 0.25))
        if near
        else 0.0,
        "retained": float(config.get("retained_credit", 0.25)) if retained else 0.0,
    }
    credit = sum(components.values())
    if retained:
        status = "primary_retained"
    elif valid:
        status = "provisional_valid"
    elif failure_kind == "fidelity_screen_not_promoted":
        status = "provisional_screened"
    else:
        status = "rejected"
    reasons = [name for name, value in components.items() if value > 0]
    assessment = {
        "schema_version": "1.0",
        "event": "developmental_assessment",
        "timestamp": utc_now(),
        "run_id": record.get("run_id"),
        "opportunity": record.get("opportunity"),
        "candidate_id": record.get("candidate_id"),
        "status": status,
        "credit": credit,
        "credit_components": components,
        "reasons": reasons,
        "selection_effect": str(config.get("selection_effect", "none")),
        "relative_objective_gap": relative_gap,
        "semantic_delta_fingerprint": fingerprint,
        "mechanism": mechanism,
        "metrics": metrics,
    }
    append_jsonl(run_dir / "events.jsonl", assessment)
    atomic_json(
        run_dir
        / "opportunities"
        / f"{int(record['opportunity']):04d}"
        / "developmental-assessment.json",
        assessment,
    )
    if status in {"primary_retained", "provisional_valid", "provisional_screened"}:
        items.append(assessment)
        capacity = int(config.get("archive_capacity", 8))
        items.sort(
            key=lambda item: (
                float(item.get("credit", 0.0)),
                int(item.get("opportunity", 0)),
            ),
            reverse=True,
        )
        items = items[:capacity]
        atomic_json(
            archive_path,
            {
                "schema_version": "1.0",
                "selection_effect": str(config.get("selection_effect", "none")),
                "items": items,
            },
        )
    return assessment
