from dataclasses import asdict, replace
from pathlib import Path

import torch

from experiments.c0c3_factorial import tiny_adderboard_v21 as evaluator
from experiments.c0c3_factorial.evaluator import (
    CommandEvaluator,
    _release_slot,
    _try_acquire_slot,
)
from experiments.c0c3_factorial.spec import (
    FactorialSpec,
    TaskSpec,
    conditions_for_protocol,
    make_assignments,
    sha256_json,
)
from experiments.c0c3_factorial.tiny_v21_runtime import ModalFallbackEvaluator

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/c0c3_factorial"
TASK = TaskSpec.from_toml(PACKAGE / "configs/tasks/tiny_adderboard_v2_1_modal.toml")


def test_twenty_trajectories_without_altering_legacy_hashes():
    legacy = FactorialSpec.from_toml(
        PACKAGE / "configs/protocols/controlled_openevolve_transformer_v2_1.toml"
    )
    legacy_payload = asdict(legacy)
    legacy_payload.pop("include_c4")
    assert legacy.protocol_hash == sha256_json(legacy_payload)
    assert len(conditions_for_protocol("2.1")) == 5
    spec = FactorialSpec.from_toml(
        PACKAGE / "configs/protocols/tiny_adderboard_v2_1.toml"
    )
    rows = make_assignments(spec, task_id=TASK.task_id, framework_id="openevolve")
    assert len(rows) == 20
    assert spec.budget.proposals == spec.budget.candidate_evaluations == 200
    assert spec.model.name == "gpt-5.6-sol" and spec.model.reasoning_effort == "xhigh"
    for block in range(1, 6):
        peers = [row for row in rows if row.block == block]
        assert {str(row.condition) for row in peers} == {"C0", "C1", "C2", "C3"}
        assert len({row.run_seed for row in peers}) == 1


def test_seed_and_editable_representation_and_steps():
    program = evaluator.data._load_program(PACKAGE / "task_sources/tiny_adderboard_v21")
    assert evaluator.data._parameter_count(program.build_model()) == 1012
    assert program.TRAINING_STEPS == 400
    assert not hasattr(program, "EVALUATION_LADDER")
    program.TRAINING_STEPS = 733
    assert evaluator._positive_setting(program, "TRAINING_STEPS") == 733
    # Unpaired tokens are valid; representation is not frozen to pair tokens.
    program.encode_inputs = lambda left, right: torch.cat((left, right), 1)
    left, right = torch.tensor([1234, 0]), torch.tensor([5678, 9999])
    prompt = evaluator._prompt(program, left, right)
    assert prompt.shape == (2, 8)
    inputs, targets = evaluator._training_tensors(program, left, right)
    assert inputs.shape == targets.shape == (2, 13)
    assert (targets[:, :7] == -100).all()


def runtime(tmp_path, **options):
    return ModalFallbackEvaluator(
        task=TASK,
        support_source=tmp_path,
        repo_root=tmp_path,
        python_bin="python",
        options=options,
        slot_root=tmp_path / "obsolete",
        max_parallel_evaluators=1,
    )


def arguments(tmp_path):
    return dict(
        candidate_snapshot=tmp_path / "candidate",
        opportunity_root=tmp_path / "op",
        timeout_seconds=20,
        run_seed=42,
    )


def test_modal_never_takes_local_or_legacy_slots(tmp_path, monkeypatch):
    instance = runtime(tmp_path)
    assert instance.slot_root is None and instance.shared_slot_root is None
    sentinel = object()
    monkeypatch.setattr(instance, "_remote", lambda **kw: sentinel)
    monkeypatch.setattr(
        CommandEvaluator,
        "evaluate",
        lambda **kw: (_ for _ in ()).throw(AssertionError()),
    )
    assert instance.evaluate(**arguments(tmp_path)) is sentinel
    assert not list(instance.fallback_root.glob("*.lock"))


def test_fallback_is_capped_and_queued_work_can_return_to_modal(tmp_path, monkeypatch):
    instance = runtime(tmp_path, modal_retry_seconds=0)
    instance.fallback_root.mkdir(parents=True)
    leases = [
        _try_acquire_slot(
            root=instance.fallback_root,
            capacity=3,
            first=0,
            opportunity_root=tmp_path,
            scope="test",
        )
        for _ in range(3)
    ]
    assert all(leases)
    assert (
        _try_acquire_slot(
            root=instance.fallback_root,
            capacity=3,
            first=0,
            opportunity_root=tmp_path,
            scope="test",
        )
        is None
    )
    sentinel = object()
    responses = iter([None, sentinel])
    monkeypatch.setattr(instance, "_remote", lambda **kw: next(responses))
    try:
        assert instance.evaluate(**arguments(tmp_path)) is sentinel
    finally:
        for lease in leases:
            _release_slot(lease)


def test_modal_failure_runs_local_and_releases_lease(tmp_path, monkeypatch):
    instance = runtime(tmp_path)
    sentinel = object()
    monkeypatch.setattr(instance, "_remote", lambda **kw: None)
    monkeypatch.setattr(CommandEvaluator, "evaluate", lambda self, **kw: sentinel)
    assert instance.evaluate(**arguments(tmp_path)) is sentinel
    lease = _try_acquire_slot(
        root=instance.fallback_root,
        capacity=3,
        first=0,
        opportunity_root=tmp_path,
        scope="test",
    )
    assert lease is not None
    _release_slot(lease)


def test_no_subject_worker_limit_is_task_scoped():
    assert TASK.extension_options["unlimited_subject_workers"] is True
    legacy = replace(TASK, extension_options={})
    assert not legacy.extension_options.get("unlimited_subject_workers", False)
