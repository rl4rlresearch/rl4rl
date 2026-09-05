import json
from pathlib import Path

import pytest

from experiments import live_trajectory_dashboard as dashboard
from experiments.c0c3_factorial import evaluator
from experiments.c0c3_factorial.spec import Condition, TaskSpec
from experiments.c0c3_factorial.state import SearchController
from experiments.c0c3_factorial.tiny_v21_runtime import ModalFallbackEvaluator
from tests.test_uci_har_pareto import SPEC, candidate, complete


def test_dashboard_plots_run_hypervolume_not_best_accuracy(tmp_path):
    c = SearchController.create(
        tmp_path / "run",
        SPEC,
        run_id="run",
        condition=Condition.C0,
        seed_candidate=candidate("seed", 0.8, 1_000_000),
    )
    complete(c, "faster", 0.7, 500_000)
    complete(c, "dominated", 0.6, 800_000)
    result = dashboard.build_run(
        tmp_path / "run",
        dashboard.DEFAULT_PRICE_PER_MILLION,
        objective_metric="hypervolume",
        objective_direction="maximize",
    )
    assert [p["best_objective"] for p in result["points"]] == pytest.approx(
        [0.4, 0.575, 0.575]
    )
    assert result["points"][1]["metrics"]["validation_accuracy"] == 0.7
    assert result["points"][1]["metrics"]["inference_macs"] == 500_000
    assert "'uci_har_pareto_v21'" in dashboard.PAGE


def test_fallback_pool_shared_between_tasks_and_checkouts(tmp_path, monkeypatch):
    monkeypatch.setenv("RL4RL_SHARED_LOCAL_EVALUATOR_ROOT", str(tmp_path / "host"))
    root = (
        Path(__file__).resolve().parents[1] / "experiments/c0c3_factorial/configs/tasks"
    )
    instances = [
        ModalFallbackEvaluator(
            task=TaskSpec.from_toml(root / config),
            support_source=tmp_path,
            repo_root=tmp_path / str(index),
            python_bin="python",
            options={},
        )
        for index, config in enumerate(
            ("tiny_adderboard_v2_1_modal.toml", "uci_har_pareto_v2_1_modal.toml")
        )
    ]
    assert (
        instances[0].fallback_root
        == instances[1].fallback_root
        == evaluator.windows_evaluator_root()
    )
    pool = instances[0].fallback_root
    pool.mkdir(parents=True)
    leases = [
        evaluator._try_acquire_slot(
            root=pool, capacity=3, first=0, opportunity_root=tmp_path, scope="test"
        )
        for _ in range(3)
    ]
    try:
        assert all(leases)
        assert (
            evaluator._try_acquire_slot(
                root=instances[1].fallback_root,
                capacity=3,
                first=0,
                opportunity_root=tmp_path,
                scope="test",
            )
            is None
        )
    finally:
        for lease in leases:
            evaluator._release_slot(lease)


def test_native_windows_evaluator_uses_same_three_slot_pool(tmp_path, monkeypatch):
    if evaluator.os.name != "nt":
        pytest.skip("Windows admission rule")
    monkeypatch.setenv("RL4RL_SHARED_LOCAL_EVALUATOR_ROOT", str(tmp_path / "host"))
    task = TaskSpec.from_toml(
        Path(__file__).resolve().parents[1]
        / "experiments/c0c3_factorial/configs/tasks/uci_har_pareto_v2_1_modal.toml"
    )
    instance = evaluator.CommandEvaluator(
        task=task, support_source=tmp_path, repo_root=tmp_path, python_bin="python"
    )
    with instance._evaluation_slot(tmp_path):
        receipt = json.loads((tmp_path / "windows-evaluator-slot.json").read_text())
        assert receipt["capacity"] == 3
        assert Path(receipt["slot"]).parent == evaluator.windows_evaluator_root()
