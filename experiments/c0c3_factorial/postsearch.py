"""Sealed Layer B review and Layer C final evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import secrets
import shutil
from dataclasses import asdict, replace
from pathlib import Path

from .analysis import RunOutcome, estimate, write_estimate
from .evaluator import CommandEvaluator
from .spec import FactorialSpec, TaskSpec
from .state import SearchController


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_runs(
    campaign: Path, spec: FactorialSpec
) -> list[tuple[dict[str, object], SearchController]]:
    schedule = _read_json(campaign / "schedule.json")
    campaign_manifest = _read_json(campaign / "campaign.json")
    primary_run_ids = {
        str(run_id)
        for run_id in campaign_manifest.get(
            "primary_run_ids",
            [assignment["run_id"] for assignment in schedule],
        )
    }
    optional_run_ids = {
        str(run_id) for run_id in campaign_manifest.get("optional_run_ids", [])
    }
    loaded: list[tuple[dict[str, object], SearchController]] = []
    for assignment in schedule:
        controller = SearchController.load(
            campaign / "runs" / str(assignment["run_id"]), spec
        )
        loaded.append((assignment, controller))

    scope_run_ids = set(primary_run_ids)
    optional_stages: dict[tuple[int, str], list[SearchController]] = {}
    for assignment, controller in loaded:
        if controller.state.run_id not in optional_run_ids:
            continue
        stage_kind = "no-search" if controller.state.no_search else "factorial"
        key = (int(assignment["block"]), stage_kind)
        optional_stages.setdefault(key, []).append(controller)
    for controllers in optional_stages.values():
        activated = any(
            controller.state.status != "ready"
            or controller.state.proposals_used > 0
            for controller in controllers
        )
        if activated:
            scope_run_ids.update(controller.state.run_id for controller in controllers)

    result = []
    incomplete = []
    for assignment, controller in loaded:
        if controller.state.run_id not in scope_run_ids:
            continue
        if controller.state.status != "completed":
            incomplete.append(controller.state.run_id)
        result.append((assignment, controller))
    if incomplete:
        raise RuntimeError(
            "post-search layers are sealed until every run completes within the "
            "activated analysis scope: "
            + ", ".join(incomplete)
        )
    return result


def _proposal_events(path: Path) -> list[dict[str, object]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "proposal_completed":
            events.append(event)
    return events


def export_layer_b_packets(
    campaign_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
) -> Path:
    campaign = Path(campaign_dir).resolve()
    runs = _completed_runs(campaign, spec)
    sealed = campaign / "sealed-layer-b"
    packets_root = sealed / "packets"
    if sealed.exists():
        raise FileExistsError("Layer B export already exists")
    packets_root.mkdir(parents=True)
    salt = secrets.token_hex(32)
    mapping: list[dict[str, object]] = []
    for assignment, controller in runs:
        run_dir = campaign / "runs" / controller.state.run_id
        for event in _proposal_events(controller.events_path):
            evaluation = event["evaluation"]
            if not isinstance(evaluation, dict) or not evaluation.get("valid"):
                continue
            opaque = hashlib.sha256(
                (
                    f"{salt}:{controller.state.run_id}:"
                    f"{event['opportunity']}:{event['candidate_id']}"
                ).encode()
            ).hexdigest()[:20]
            packet_id = f"packet-{opaque}"
            source = run_dir / str(event["artifact_path"])
            destination = packets_root / packet_id / "candidate"
            shutil.copytree(source, destination)
            parent_ids = event["parent_ids"]
            if not isinstance(parent_ids, list) or len(parent_ids) != 1:
                raise ValueError("Layer B v1 requires exactly one recorded parent")
            parent_source = run_dir / "candidates" / str(parent_ids[0])
            shutil.copytree(parent_source, packets_root / packet_id / "parent")
            packet = {
                "schema_version": "1.0",
                "packet_id": packet_id,
                "task_id": task.task_id,
                "hypothesis": event["hypothesis"],
                "intended_edit": event["intended_edit"],
                "parent_count": 1,
                "layer_a_qualified": True,
                "review_question": (
                    "Does this candidate instantiate a coherent, testable mechanism "
                    "change rather than only a parameter/hyperparameter adjustment?"
                ),
            }
            (packets_root / packet_id / "packet.json").write_text(
                json.dumps(packet, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            mapping.append(
                {
                    "packet_id": packet_id,
                    "run_id": controller.state.run_id,
                    "block": assignment["block"],
                    "condition": assignment["condition"],
                    "opportunity": event["opportunity"],
                    "candidate_id": event["candidate_id"],
                }
            )
    random.Random(salt).shuffle(mapping)
    with (sealed / "packet_order.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("packet_id",), delimiter="\t")
        writer.writeheader()
        for row in mapping:
            writer.writerow({"packet_id": row["packet_id"]})
    (sealed / "annotations.template.tsv").write_text(
        "packet_id\tlayer_b_qualified\tmechanism_cluster\tprimary_mechanism"
        "\treviewer\tnotes\n",
        encoding="utf-8",
    )
    private = sealed / "private"
    private.mkdir()
    (private / "mapping.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (private / "salt.txt").write_text(salt + "\n", encoding="utf-8")
    (sealed / "scope.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_ids": [controller.state.run_id for _, controller in runs],
                "scope": "campaign_activated_analysis_scope",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (sealed / "README.md").write_text(
        "# Blinded Layer B packets\n\n"
        "Review packets in `packet_order.tsv` order. Copy "
        "`annotations.template.tsv`, add exactly one row per packet, use `1` only "
        "for a qualified mechanism change, and assign the same stable "
        "`mechanism_cluster` label to mechanistically equivalent candidates. "
        "Compare `parent/` with `candidate/`; Layer A scores are intentionally "
        "hidden from reviewers to reduce performance-outcome bias. "
        "Do not open `private/` until adjudication is frozen.\n",
        encoding="utf-8",
    )
    return sealed


def score_layer_b(
    campaign_dir: str | Path,
    *,
    annotations_path: str | Path,
) -> Path:
    campaign = Path(campaign_dir).resolve()
    sealed = campaign / "sealed-layer-b"
    mapping = _read_json(sealed / "private/mapping.json")
    with Path(annotations_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = {str(row["packet_id"]) for row in mapping}
    actual = [str(row.get("packet_id", "")) for row in rows]
    if len(actual) != len(set(actual)):
        raise ValueError("annotations contain duplicate packet IDs")
    if set(actual) != expected:
        raise ValueError(
            f"annotation packet mismatch; missing={sorted(expected - set(actual))}, "
            f"extra={sorted(set(actual) - expected)}"
        )
    annotations = {str(row["packet_id"]): row for row in rows}
    scope_run_ids = {
        str(run_id) for run_id in _read_json(sealed / "scope.json")["run_ids"]
    }
    clusters: dict[str, set[str]] = {}
    for private in mapping:
        packet_id = str(private["packet_id"])
        annotation = annotations[packet_id]
        qualified = annotation.get("layer_b_qualified")
        if qualified not in {"0", "1"}:
            raise ValueError(f"{packet_id} has invalid layer_b_qualified")
        if qualified == "1":
            cluster = str(annotation.get("mechanism_cluster", "")).strip()
            if not cluster:
                raise ValueError(f"{packet_id} is qualified without a cluster")
            clusters.setdefault(str(private["run_id"]), set()).add(cluster)
    schedule = _read_json(campaign / "schedule.json")
    output = sealed / "scored"
    output.mkdir(exist_ok=False)
    fieldnames = (
        "block",
        "condition",
        "run_id",
        "qualified_mechanism_clusters",
    )
    factorial_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    for assignment in schedule:
        if str(assignment["run_id"]) not in scope_run_ids:
            continue
        row = {
            "block": assignment["block"],
            "condition": assignment["condition"],
            "run_id": assignment["run_id"],
            "qualified_mechanism_clusters": len(
                clusters.get(str(assignment["run_id"]), set())
            ),
        }
        (baseline_rows if assignment["condition"] == "N0" else factorial_rows).append(
            row
        )
    for name, values in (
        ("factorial_outcomes.tsv", factorial_rows),
        ("no_search_outcomes.tsv", baseline_rows),
    ):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(values)
    outcomes = [
        RunOutcome(
            block=str(row["block"]),
            condition=row_condition(str(row["condition"])),
            qualified_mechanism_clusters=int(row["qualified_mechanism_clusters"]),
            run_id=str(row["run_id"]),
        )
        for row in factorial_rows
    ]
    write_estimate(output / "factorial_estimates.json", estimate(outcomes))
    return output


def row_condition(value: str):
    from .spec import Condition

    return Condition(value)


def _layer_a_selected_candidate_id(controller: SearchController) -> str:
    """Select the final candidate without adding adaptive N0 feedback.

    Factorial trajectories already maintain an online incumbent. N0 proposals
    must remain independent during search, but after all proposals finish it is
    valid to choose their best Layer-A-qualified candidate for a final evaluation.
    """

    if not controller.state.no_search:
        return controller.state.incumbent_id
    return max(
        controller.state.candidates.values(),
        key=lambda candidate: (
            candidate.fitness,
            -candidate.created_opportunity,
            candidate.candidate_id,
        ),
    ).candidate_id


def run_layer_c(
    campaign_dir: str | Path,
    *,
    spec: FactorialSpec,
    task: TaskSpec,
    repo_root: Path,
    python_bin: str,
) -> Path:
    campaign = Path(campaign_dir).resolve()
    runs = _completed_runs(campaign, spec)
    output = campaign / "sealed-layer-c"
    if output.exists():
        raise FileExistsError("Layer C output already exists")
    output.mkdir()
    final_task = replace(task, evaluator_command=task.final_holdout_command)
    summary = []
    for assignment, controller in runs:
        run_dir = campaign / "runs" / controller.state.run_id
        selected_candidate_id = _layer_a_selected_candidate_id(controller)
        evaluator = CommandEvaluator(
            task=final_task,
            support_source=run_dir / "task-support",
            repo_root=repo_root,
            python_bin=python_bin,
        )
        artifacts = evaluator.evaluate(
            candidate_snapshot=(run_dir / "candidates" / selected_candidate_id),
            opportunity_root=output / controller.state.run_id,
            timeout_seconds=spec.budget.evaluator_timeout_seconds,
            run_seed=int(assignment["run_seed"]),
        )
        summary.append(
            {
                "run_id": controller.state.run_id,
                "block": assignment["block"],
                "condition": assignment["condition"],
                "selected_by_layer_a_candidate_id": selected_candidate_id,
                "selection_rule": (
                    "post_search_best_independent_layer_a_candidate"
                    if controller.state.no_search
                    else "online_incumbent"
                ),
                "layer_c_evaluation": asdict(artifacts.evaluation),
            }
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
