from types import SimpleNamespace

import pytest

from experiments import resume_trajectory_to_target as bounded


def test_exact_pause_is_requested_before_next_opportunity(monkeypatch, tmp_path):
    state = SimpleNamespace(condition="C0", active=None, proposals_used=48)
    spec = SimpleNamespace(budget=SimpleNamespace(proposals=200))
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
