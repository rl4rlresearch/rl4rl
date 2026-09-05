from types import SimpleNamespace

import pytest

from experiments import resume_trajectory_to_target as bounded


def test_legacy_nanogpt_payload_omits_only_unused_extensions():
    from dataclasses import dataclass, field

    @dataclass
    class Task:
        adapter: str = bounded.hybrid_evaluator.NANOGPT_TASK_ADAPTER
        extension_module: str | None = None
        extension_options: dict = field(default_factory=dict)
        task_id: str = "original-task"

    assert bounded.legacy_nanogpt_task_payload(Task()) == {
        "adapter": bounded.hybrid_evaluator.NANOGPT_TASK_ADAPTER,
        "task_id": "original-task",
    }
    other = bounded.legacy_nanogpt_task_payload(Task(adapter="other"))
    assert "extension_module" in other and "extension_options" in other
    with pytest.raises(ValueError, match="cannot execute task extensions"):
        bounded.legacy_nanogpt_task_payload(Task(extension_module="required"))
    with pytest.raises(ValueError, match="cannot execute task extensions"):
        bounded.legacy_nanogpt_task_payload(Task(extension_options={"required": 1}))


@pytest.mark.parametrize("legacy", [False, True])
def test_exact_pause_is_requested_before_next_opportunity(
    monkeypatch, tmp_path, legacy
):
    state = SimpleNamespace(condition="C0", active=None, proposals_used=48)
    spec = SimpleNamespace(budget=SimpleNamespace(proposals=200))
    if legacy:
        import json

        spec.protocol_version = "2.1"
        spec.include_c4 = None
        spec.blocks = 1
        (tmp_path / "schedule.json").write_text(
            json.dumps(
                [
                    {"block": 1, "condition": condition}
                    for condition in ("C0", "C1", "C2", "C3")
                ]
            )
        )
    monkeypatch.setattr(bounded, "_load_campaign", lambda _: (spec, None, None))
    monkeypatch.setattr(
        bounded.SearchController, "load", lambda *a: SimpleNamespace(state=state)
    )
    trace = []

    def step(*a, **k):
        state.proposals_used += 1
        trace.append(state.proposals_used)
        return {"proposals_cumulative": state.proposals_used}

    monkeypatch.setattr(bounded.orchestration, "run_one_opportunity", step)
    monkeypatch.setattr(
        bounded.orchestration,
        "request_staged_trajectory_pause",
        lambda *a, **k: trace.append("pause"),
    )

    def loop(*a, **k):
        if legacy:
            assert len(bounded.orchestration.conditions_for_protocol("2.1", None)) == 4
        bounded.orchestration.run_one_opportunity(tmp_path)
        assert trace == [49]
        bounded.orchestration.run_one_opportunity(tmp_path)
        assert trace == [49, 50, "pause"]
        with pytest.raises(RuntimeError, match="beyond the target"):
            bounded.orchestration.run_one_opportunity(tmp_path)
        return {"status": "paused", "proposals_used": 50}

    monkeypatch.setattr(bounded.orchestration, "run_staged_individual_trajectory", loop)
    result = bounded.resume_to_target(
        tmp_path,
        "run",
        50,
        repo_root=tmp_path,
        python_bin="python",
        codex_binary="codex",
    )
    assert result["proposals_used"] == 50
    assert bounded.orchestration.run_one_opportunity is step


@pytest.mark.parametrize(
    "condition,count,active", [("C4", 48, None), ("C0", 50, None), ("C0", 48, object())]
)
def test_rejects_excluded_completed_or_interrupted_run(
    monkeypatch, tmp_path, condition, count, active
):
    state = SimpleNamespace(condition=condition, proposals_used=count, active=active)
    monkeypatch.setattr(
        bounded,
        "_load_campaign",
        lambda _: (SimpleNamespace(budget=SimpleNamespace(proposals=200)), None, None),
    )
    monkeypatch.setattr(
        bounded.SearchController, "load", lambda *a: SimpleNamespace(state=state)
    )
    with pytest.raises(ValueError):
        bounded.resume_to_target(
            tmp_path,
            "run",
            50,
            repo_root=tmp_path,
            python_bin="python",
            codex_binary="codex",
        )
