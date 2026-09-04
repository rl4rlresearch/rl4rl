from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from experiments.c0c3_factorial.spec import (
    FactorialSpec,
    FrameworkSpec,
    TaskSpec,
    make_assignments,
)
from experiments.c0c3_factorial.tiny_kws_rnn import (
    _MacCounter,
    _recurrent_contract,
    _run_sequence,
    _speaker_split,
    preflight_candidate_source,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/c0c3_factorial"
SEED = PACKAGE / "task_sources/tiny_kws_rnn/train.py"


def _seed_module():
    spec = importlib.util.spec_from_file_location("tiny_kws_test_seed", SEED)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_is_genuinely_recurrent_and_exactly_counted() -> None:
    model = _seed_module().build_model()
    frames = torch.randn(3, 32, 20)
    contract = _recurrent_contract(model, frames)
    assert contract["state_dependence_effect"] > 0
    counter = _MacCounter(model)
    logits, recurrent_steps, _peak, depths = _run_sequence(
        model, frames, counter=counter
    )
    counter.close()
    assert logits.shape == (3, 8)
    assert recurrent_steps == 3 * 32
    assert depths == [32, 32, 32]
    assert counter.macs > counter.classifier_macs > 0


def test_source_preflight_rejects_uncounted_matrix_multiplication(
    tmp_path: Path,
) -> None:
    (tmp_path / "train.py").write_text(
        SEED.read_text(encoding="utf-8").replace(
            "return self.classifier(summary / count.clamp_min(1.0))",
            "return summary @ self.classifier.weight.T",
        ),
        encoding="utf-8",
    )
    assert "uncounted matrix multiplication" in str(
        preflight_candidate_source(tmp_path)
    )


def test_speaker_hash_split_is_deterministic_and_nonempty() -> None:
    values = {_speaker_split(f"speaker-{index}") for index in range(1000)}
    assert values == {"train", "validation", "layer_c"}
    assert _speaker_split("speaker-17") == _speaker_split("speaker-17")


def test_v21_tiny_kws_configuration_declares_five_c0_c4_blocks() -> None:
    protocol = FactorialSpec.from_toml(
        PACKAGE / "configs/protocols/tiny_kws_rnn_openevolve_v2_1.toml"
    )
    task = TaskSpec.from_toml(
        PACKAGE / "configs/tasks/tiny_kws_rnn_source_only_cpu.toml"
    )
    framework = FrameworkSpec.from_toml(
        PACKAGE / "configs/frameworks/openevolve_tiny_kws_rnn_v2_1.toml"
    )
    assignments = make_assignments(
        protocol, task_id=task.task_id, framework_id=framework.framework_key
    )
    assert len(assignments) == 25
    assert {assignment.condition.value for assignment in assignments} == {
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
    }
    assert protocol.model.name == "gpt-5.6-sol"
    assert protocol.model.reasoning_effort == "xhigh"
