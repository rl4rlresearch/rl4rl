import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.c0c3_factorial import pareto, postsearch, uci_har
from experiments.c0c3_factorial.spec import (
    Condition,
    FactorialSpec,
    TaskSpec,
    make_assignments,
)
from experiments.c0c3_factorial.state import (
    Candidate,
    Evaluation,
    SearchController,
    Usage,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/c0c3_factorial"
SPEC = FactorialSpec.from_toml(PACKAGE / "configs/protocols/uci_har_pareto_v2_1.toml")
TASK = TaskSpec.from_toml(PACKAGE / "configs/tasks/uci_har_pareto_v2_1_modal.toml")


def candidate(name, accuracy, cost, index=0):
    return Candidate(
        name,
        [],
        accuracy,
        {"validation_accuracy": accuracy, "inference_macs": cost},
        name,
        "test",
        "test",
        index,
        index,
    )


def test_hypervolume_union_dominance_and_ties():
    a = candidate("a", 0.8, 1_000_000)
    b = candidate("b", 0.9, 1_500_000, 1)
    c = candidate("c", 0.7, 1_200_000, 2)
    tie = candidate("tie", 0.8, 1_000_000, 3)
    assert pareto.hypervolume([a]) == pytest.approx(0.4)
    assert pareto.hypervolume([a, b, c, tie]) == pytest.approx(0.425)
    assert [p.candidate_id for p in pareto.frontier([a, b, c, tie])] == ["a", "b"]
    assert pareto.hypervolume([candidate("limit", 1, pareto.MAX_MACS)]) == 0
    for cost in (0, -1, float("nan"), 2_000_001):
        with pytest.raises(ValueError):
            pareto.point({"validation_accuracy": 0.9, "inference_macs": cost})


def complete(controller, name, accuracy, cost):
    controller.begin()
    return controller.complete(
        candidate_id=name,
        artifact_path=name,
        hypothesis="test",
        intended_edit="test",
        evaluation=Evaluation(
            True,
            accuracy,
            {"validation_accuracy": accuracy, "inference_macs": cost},
            0.01,
        ),
        usage=Usage(),
        prompt_hashes={},
    )


def test_discarded_frontier_point_blocks_a_later_child_and_chart_survives_reload(
    tmp_path,
):
    c = SearchController.create(
        tmp_path / "run",
        SPEC,
        run_id="run",
        condition=Condition.C0,
        seed_candidate=candidate("seed", 0.8, 1_000_000),
    )
    first = complete(c, "faster", 0.7, 500_000)
    assert first["retained"] and c.state.incumbent_id == "faster"
    # Seed is no longer visible but must still block this more expensive child.
    second = complete(c, "dominated_by_discarded_seed", 0.79, 1_100_000)
    assert not second["retained"]
    assert second["pareto"]["hypervolume"] == first["pareto"]["hypervolume"]
    c = SearchController.load(tmp_path / "run", SPEC)
    third = complete(c, "accurate", 0.9, 1_500_000)
    assert third["retained"]
    assert third["pareto"]["hypervolume"] > first["pareto"]["hypervolume"]
    assert len(c.state.candidates) == 4
    assert c.state.portfolio_ids == ["accurate"]


def test_portfolio_capacity_parent_selection_and_non_scalar_admission(tmp_path):
    c = SearchController.create(
        tmp_path / "run",
        SPEC,
        run_id="run",
        condition=Condition.C2,
        seed_candidate=candidate("seed", 0.8, 1_000_000),
    )
    for i, (accuracy, cost) in enumerate(
        [(0.7, 500_000), (0.9, 1_500_000), (0.6, 250_000), (0.95, 1_750_000)]
    ):
        record = complete(c, str(i), accuracy, cost)
        assert record["retained"]
        assert len(c.state.portfolio_ids) <= 4
    assert len(c.state.portfolio_ids) == 4
    assert record["retention_decision"] == "archive_pareto_replaced_selected_lineage"
    assert len(record["pareto"]["frontier_ids"]) == 5


def test_configuration_and_seed_identity():
    rows = make_assignments(SPEC, task_id=TASK.task_id, framework_id="openevolve")
    assert len(rows) == 20
    assert {str(row.condition) for row in rows} == {"C0", "C1", "C2", "C3"}
    assert SPEC.blocks == 5 and SPEC.budget.proposals == 200
    assert SPEC.budget.evaluator_timeout_seconds == 240
    assert TASK.extension_options["unlimited_subject_workers"] is True
    assert TASK.qualification_metric is None
    with pytest.raises(ValueError):
        replace(
            SPEC,
            parent_selection_rule="fill_from_seed_then_least_selected_lineage_then_best_then_oldest_then_id_v1",
        )
    program = uci_har.load_program(PACKAGE / "task_sources/uci_har")
    model = program.build_model().eval()
    assert sum(p.numel() for p in model.parameters()) == 10454
    counter = uci_har.MacCounter(model)
    try:
        with torch.no_grad(), counter:
            logits = model(torch.randn(2, 128, 9))
        assert logits.shape == (2, 6)
        assert counter.macs == 2 * 420128
    finally:
        counter.close()


def test_counter_rejects_uncounted_functional_affine_and_subclasses():
    class Functional(nn.Module):
        def forward(self, x):
            return x @ torch.ones(9, 6)

    model = Functional()
    counter = uci_har.MacCounter(model)
    with pytest.raises(ValueError, match="counted nn primitives"), counter:
        model(torch.ones(2, 9))

    class Disguised(nn.Linear):
        pass

    with pytest.raises(ValueError, match="No MAC rule"):
        uci_har.MacCounter(Disguised(9, 6))
    patched = nn.Linear(9, 6)
    patched.forward = lambda x: x[:, :6]
    with pytest.raises(ValueError, match="override forward"):
        uci_har.MacCounter(patched)


def test_bidirectional_multilayer_macs_and_grouped_convolution():
    model = nn.Sequential(nn.Conv1d(4, 4, 3, groups=2), nn.Flatten(), nn.Linear(24, 6))
    counter = uci_har.MacCounter(model)
    try:
        with counter:
            model(torch.ones(1, 4, 8))
        assert counter.macs == 24 * 2 * 3 + 24 * 6
    finally:
        counter.close()
    model = nn.LSTM(3, 5, num_layers=2, bidirectional=True, batch_first=True)
    counter = uci_har.MacCounter(model)
    try:
        with counter:
            model(torch.ones(2, 7, 3))
        assert counter.macs == 2 * 7 * 2 * 4 * 5 * ((3 + 5) + (10 + 5))
    finally:
        counter.close()


def test_modified_manifest_cannot_redefine_the_validation_split(tmp_path):
    manifest = {"splits": {"validation": {"sha256": "changed"}}}
    with pytest.raises(ValueError, match="checksum"):
        uci_har.read_split(tmp_path, "validation", manifest)


def test_invalid_evaluation_consumes_budget_without_changing_frontier(tmp_path):
    c = SearchController.create(
        tmp_path / "run",
        SPEC,
        run_id="run",
        condition=Condition.C0,
        seed_candidate=candidate("seed", 0.8, 1_000_000),
    )
    c.begin()
    event = c.complete(
        candidate_id="failed",
        artifact_path="failed",
        hypothesis="test",
        intended_edit="test",
        evaluation=Evaluation(False, None, {}, 0.01),
        usage=Usage(),
        prompt_hashes={},
    )
    assert not event["retained"]
    assert event["pareto"]["hypervolume_increment"] == 0
    assert event["pareto"]["frontier_ids"] == ["seed"]
    assert c.state.proposals_used == c.state.evaluations_used == 1
    assert c.state.active is None


@pytest.mark.parametrize("fail_one", [False, True])
def test_holdout_covers_discarded_frontier_and_marks_partial_results_incomplete(
    tmp_path,
    monkeypatch,
    fail_one,
):
    c = SearchController.create(
        tmp_path / "run",
        SPEC,
        run_id="run",
        condition=Condition.C0,
        seed_candidate=candidate("seed", 0.8, 1_000_000),
    )
    complete(c, "faster", 0.7, 500_000)
    complete(c, "dominated", 0.6, 800_000)
    assert c.state.portfolio_ids == ["faster"]
    monkeypatch.setattr(
        postsearch,
        "_completed_runs",
        lambda *_: [
            (
                {"block": 1, "condition": "C0", "run_seed": 42},
                c,
            )
        ],
    )
    called = []

    def evaluate(**kwargs):
        name = kwargs["candidate_snapshot"].name
        called.append(name)
        assert kwargs["run_seed"] == 42
        valid = not (fail_one and name == "seed")
        return SimpleNamespace(
            evaluation=Evaluation(
                valid,
                c.state.candidates[name].fitness if valid else None,
                c.state.candidates[name].metrics if valid else {},
                0.01,
            )
        )

    def make_evaluator(**kwargs):
        assert "--holdout" in kwargs["task"].evaluator_command
        return SimpleNamespace(evaluate=evaluate)

    monkeypatch.setattr(postsearch, "make_command_evaluator", make_evaluator)
    output = postsearch._run_layer_c_unlocked(
        tmp_path,
        spec=SPEC,
        task=TASK,
        repo_root=ROOT,
        python_bin="python",
    )
    result = json.loads((output / "summary.json").read_text())[0]
    assert called == ["seed", "faster"]
    assert result["complete"] is not fail_one
    if fail_one:
        assert result["holdout_hypervolume"] is None
    else:
        assert result["holdout_hypervolume"] == pytest.approx(0.575)
