"""Structural and training-boundary checks for the compact seed.

Accuracy is verified separately with fresh training and the protected evaluator.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = ROOT / "experiments/c0c3_factorial/task_sources/tiny_adderboard_compact"


@pytest.fixture(scope="module")
def program():
    torch = pytest.importorskip("torch")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location(
        "compact_seed_test", SEED / "train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    torch.set_num_threads(previous_threads)


def test_compact_seed_passes_protected_source_preflight():
    from experiments.c0c3_factorial.tiny_adderboard import preflight_candidate_source

    assert preflight_candidate_source(SEED) is None


def test_future_tiny_adderboard_task_uses_compact_seed():
    import tomllib

    path = (
        ROOT
        / "experiments/c0c3_factorial/configs/tasks"
        / "tiny_adderboard_semantic_v4_mps.toml"
    )
    task = tomllib.loads(path.read_text(encoding="utf-8"))
    assert ROOT / task["seed_source"] == SEED


def test_compact_parameters_are_independent_and_trainable(program):
    model = program.build_model()
    parameters = list(model.named_parameters(remove_duplicate=False))
    assert sum(p.numel() for _, p in parameters) == 1_012
    assert all(p.requires_grad and p.numel() > 0 for _, p in parameters)
    assert len({id(p) for _, p in parameters}) == len(parameters)
    assert len({p.untyped_storage().data_ptr() for _, p in parameters}) == len(
        parameters
    )
    assert not list(model.buffers())


def test_compact_model_initializes_fresh_and_preserves_causality(program):
    torch = pytest.importorskip("torch")
    with torch.random.fork_rng():
        torch.manual_seed(19)
        model = program.build_model().eval()
        torch.manual_seed(23)
        other = program.build_model().eval()
        assert any(
            not torch.equal(left, right)
            for left, right in zip(model.parameters(), other.parameters(), strict=True)
        )
        tokens = torch.randint(0, 114, (3, 11))
        changed = tokens.clone()
        changed[:, 6:] = (changed[:, 6:] + 17) % 114
        with torch.no_grad():
            logits = model(tokens)
            changed_logits = model(changed)
            prefix_logits = model(tokens[:, :6])
        assert logits.shape == (3, 11, 114)
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(logits[:, :6], changed_logits[:, :6])
        torch.testing.assert_close(logits[:, :6], prefix_logits)


def test_only_optimizer_step_updates_parameters_and_covers_each_once(program):
    torch = pytest.importorskip("torch")
    with torch.random.fork_rng():
        torch.manual_seed(31)
        model = program.build_model()
        optimizer = program.build_optimizer(model, 1_000)
        optimized = [p for group in optimizer.param_groups for p in group["params"]]
        assert len({id(p) for p in optimized}) == len(optimized)
        assert {id(p) for p in optimized} == {id(p) for p in model.parameters()}
        tokens = torch.randint(0, 114, (8, 11))
        targets = torch.randint(0, 114, (8, 11))
        targets[:, :5] = -100
        before = [p.detach().clone() for p in model.parameters()]
        loss = program.training_loss(model, tokens, targets, 1, 1_000)
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert all(
            torch.equal(old, current)
            for old, current in zip(before, model.parameters(), strict=True)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        routing = model.block.attention.relative_bias.weight
        assert routing.grad is not None
        assert torch.isfinite(routing.grad).all() and routing.grad.abs().sum() > 0
        optimizer.step()
        after = [p.detach().clone() for p in model.parameters()]
        assert any(
            not torch.equal(old, current)
            for old, current in zip(before, after, strict=True)
        )
        program.after_optimizer_step(optimizer, 1, 1_000)
        assert all(
            torch.equal(old, current)
            for old, current in zip(after, model.parameters(), strict=True)
        )
