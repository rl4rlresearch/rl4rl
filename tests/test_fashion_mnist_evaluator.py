from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.c0c3_factorial.fashion_mnist as fashion_mnist  # noqa: E402

SEED_SOURCE = (
    ROOT / "experiments/c0c3_factorial/task_sources/fashion_mnist/train.py"
)


@pytest.mark.parametrize(
    ("layer", "expected_split", "expected_cases"),
    (("A", "public_validation", 10), ("C", "sealed_test", 7)),
)
def test_protected_fashion_mnist_loop_enforces_exposure_and_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layer: str,
    expected_split: str,
    expected_cases: int,
) -> None:
    torch = pytest.importorskip("torch")
    train_images = torch.zeros((60, 28, 28), dtype=torch.uint8)
    train_labels = torch.arange(60, dtype=torch.long).remainder(10)
    test_images = torch.ones((7, 28, 28), dtype=torch.uint8)
    test_labels = torch.arange(7, dtype=torch.long).remainder(10)
    monkeypatch.setattr(fashion_mnist, "TRAIN_EXAMPLES", 50)
    monkeypatch.setattr(
        fashion_mnist,
        "load_dataset",
        lambda _root: (train_images, train_labels, test_images, test_labels),
    )
    monkeypatch.setattr(
        fashion_mnist,
        "frozen_split_indices",
        lambda: (torch.arange(50), torch.arange(50, 60)),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copy2(SEED_SOURCE, workspace / "train.py")
    output = tmp_path / "evaluation.json"
    args = argparse.Namespace(
        workspace=workspace,
        repo_root=ROOT,
        output=output,
        data_root=None,
        device="cpu",
        training_examples=100,
        max_parameters=250_000,
        evaluation_batch_size=16,
        seed=1234,
        layer=layer,
    )

    assert fashion_mnist.evaluate(args) == 0
    metrics = json.loads(output.read_text())["metrics"]
    assert metrics["examples_processed"] == 100
    assert metrics["optimizer_steps"] == 2
    assert metrics["parameters"] == 105_866
    assert metrics["evaluation_split"] == expected_split
    assert metrics["evaluation_cases"] == expected_cases


def test_fashion_ladder_uses_one_training_trajectory_with_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    train_images = torch.zeros((60, 28, 28), dtype=torch.uint8)
    train_labels = torch.arange(60, dtype=torch.long).remainder(10)
    test_images = torch.ones((7, 28, 28), dtype=torch.uint8)
    test_labels = torch.arange(7, dtype=torch.long).remainder(10)
    monkeypatch.setattr(fashion_mnist, "TRAIN_EXAMPLES", 50)
    monkeypatch.setattr(
        fashion_mnist,
        "load_dataset",
        lambda _root: (train_images, train_labels, test_images, test_labels),
    )
    monkeypatch.setattr(
        fashion_mnist,
        "frozen_split_indices",
        lambda: (torch.arange(50), torch.arange(50, 60)),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = SEED_SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        "EVALUATION_LADDER = [25_000, 50_000, 100_000]",
        "EVALUATION_LADDER = [25, 50, 100]",
    ).replace(
        "EVALUATION_PROMOTION_THRESHOLDS = [0.82, 0.87, None]",
        "EVALUATION_PROMOTION_THRESHOLDS = [0.0, 0.0, None]",
    )
    (workspace / "train.py").write_text(source, encoding="utf-8")
    output = tmp_path / "evaluation.json"
    args = argparse.Namespace(
        workspace=workspace,
        repo_root=ROOT,
        output=output,
        data_root=None,
        device="cpu",
        training_examples=100,
        max_parameters=250_000,
        evaluation_batch_size=16,
        seed=1234,
        layer="A",
        multi_fidelity=True,
        ladder_levels=[25, 50, 100],
        promotion_thresholds=[0.0, 0.0, None],
        ladder_minimum=10,
        ladder_max_rungs=6,
    )

    assert fashion_mnist.evaluate(args) == 0
    payload = json.loads(output.read_text())
    metrics = payload["metrics"]
    assert payload["valid"] is True
    assert metrics["examples_processed"] == 100
    assert metrics["optimizer_steps"] == 2
    assert metrics["fidelity_reached_full"] is True
    assert metrics["fidelity_stage_count"] == 3
    assert metrics["fidelity_policy"]["accepted"] is True
    assert [
        stage["actual_examples_processed"] for stage in metrics["fidelity_stages"]
    ] == [50, 50, 100]
