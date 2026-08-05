import json
from types import SimpleNamespace

import pytest

from scripts import study_development_run
from scripts.study_development_run import main


class _FakeCompletions:
    def create(self, **_request):
        response = (
            "Hypothesis: a slightly wider feedforward path may improve digit-state "
            "transformation.\n"
            "<<<<<<< SEARCH\n"
            "        ff_dim=48,\n"
            "=======\n"
            "        ff_dim=56,\n"
            ">>>>>>> REPLACE\n"
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=17),
        )


class _FakeOpenAI:
    def __init__(self, **_configuration):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_development_dry_run_is_provider_free_and_bounded(tmp_path, capsys):
    assert (
        main(
            [
                "--study-id",
                "test-development-run",
                "--output-root",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["scientific"] is False
    assert payload["dry_run"] is True
    assert payload["cost_guard"]["maximum_provider_calls"] == 4
    assert payload["cost_guard"]["maximum_completion_tokens_across_study"] == 4_800
    assert len(payload["run_order"]) == 4
    assert not any(tmp_path.iterdir())


def test_development_run_rejects_design_above_call_ceiling(tmp_path):
    with pytest.raises(SystemExit, match="above --max-provider-calls"):
        main(
            [
                "--study-id",
                "too-many-calls",
                "--output-root",
                str(tmp_path),
                "--opportunities",
                "2",
                "--transition-opportunities",
                "1,2",
                "--dry-run",
            ]
        )


def test_development_entrypoint_completes_with_fake_provider_and_real_cpu_training(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(study_development_run, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    assert (
        main(
            [
                "--study-id",
                "fake-provider-development-run",
                "--output-root",
                str(tmp_path),
                "--device",
                "cpu",
                "--allow-cpu-for-tests",
                "--max-completion-tokens",
                "128",
                "--confirm-paid-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["scientific"] is False
    assert payload["scheduler"]["counts"]["completed"] == 4
    assert payload["totals"]["provider_attempts"] == 4
    assert payload["totals"]["candidate_training_attempts"] == 8
    assert payload["totals"]["training_steps"] == 80
    assert payload["totals"]["evaluation_cases"] == 512
    assert all(run["accepted"] == 1 for run in payload["runs"])
