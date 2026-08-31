# ruff: noqa: E402 -- repository root is inserted for standalone test runs.

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.c0c3_factorial.neutral_task import validate_v15_pairing
from experiments.c0c3_factorial.prompts import (
    PromptContext,
    PromptRenderer,
    VisibleCandidate,
    neutral_disclosure_terms,
)
from experiments.c0c3_factorial.spec import (
    Condition,
    FactorialSpec,
    FrameworkSpec,
    TaskSpec,
)
from experiments.c0c3_factorial.tiny_adderboard import (
    HOLDOUT_BUCKET_START,
    HOLDOUT_BUCKET_STOP,
    PUBLIC_BUCKET_START,
    PUBLIC_BUCKET_STOP,
    TRAIN_BUCKET_STOP,
    _split_bucket,
    preflight_candidate_source,
)

FACTORIAL = ROOT / "experiments/c0c3_factorial"
PROTOCOL = (
    FACTORIAL
    / "configs/protocols/semantic_interventions_v4_tiny_adderboard_terra.toml"
)
TASK = FACTORIAL / "configs/tasks/tiny_adderboard_semantic_v4_mps.toml"
FRAMEWORK = FACTORIAL / "configs/frameworks/tiny_adderboard_openevolve_v4.toml"
SEED = FACTORIAL / "task_sources/tiny_adderboard"


def test_tiny_adderboard_configs_are_v4_terra_xhigh() -> None:
    spec = FactorialSpec.from_toml(PROTOCOL)
    task = TaskSpec.from_toml(TASK)
    framework = FrameworkSpec.from_toml(FRAMEWORK)

    validate_v15_pairing(
        protocol_version=spec.protocol_version,
        task_adapter=task.adapter,
        prompt_profile=framework.prompt_profile,
    )
    assert spec.protocol_version == "3.0"
    assert spec.model.name == "gpt-5.6-terra"
    assert spec.model.reasoning_effort == "xhigh"
    assert spec.budget.proposals == 200
    assert task.qualification_metric == "accuracy"
    assert task.qualification_minimum == 0.99
    assert task.objective_metric == "parameters"


def test_tiny_adderboard_seed_passes_source_preflight(tmp_path: Path) -> None:
    assert preflight_candidate_source(SEED) is None
    source = (SEED / "train.py").read_text(encoding="utf-8")
    (tmp_path / "train.py").write_text(
        source + "\n# carry-specific lookup_table\n", encoding="utf-8"
    )
    assert preflight_candidate_source(tmp_path) is not None


def test_tiny_adderboard_hash_partitions_are_disjoint() -> None:
    sets = [set(), set(), set()]
    for left in range(0, 10_000, 79):
        for right in range(0, 10_000, 83):
            pair = (left, right)
            bucket = _split_bucket(left, right)
            if bucket < TRAIN_BUCKET_STOP:
                sets[0].add(pair)
            elif PUBLIC_BUCKET_START <= bucket < PUBLIC_BUCKET_STOP:
                sets[1].add(pair)
            elif HOLDOUT_BUCKET_START <= bucket < HOLDOUT_BUCKET_STOP:
                sets[2].add(pair)
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])


def test_tiny_adderboard_subject_prompt_is_clean_and_task_specific() -> None:
    spec = FactorialSpec.from_toml(PROTOCOL)
    task = TaskSpec.from_toml(TASK)
    framework = FrameworkSpec.from_toml(FRAMEWORK)
    renderer = PromptRenderer(FACTORIAL / "templates")
    prompt = renderer.render(
        spec,
        task,
        framework,
        PromptContext(
            condition=Condition.C1,
            opportunity=6,
            selected_parent_id="seed",
            visible_candidates=(
                VisibleCandidate(
                    "seed",
                    -1_548.0,
                    {"accuracy": 1.0, "parameters": 1_548},
                    0,
                    "current",
                ),
            ),
            remaining_proposals=195,
            remaining_evaluations=195,
            remaining_tokens=10**12,
            remaining_evaluator_seconds=1_000,
        ),
    )
    assert "four-digit" in prompt.text
    assert "one uninterrupted trajectory" in prompt.text
    assert "99%" in prompt.text
    assert prompt.transition_active
    assert neutral_disclosure_terms(prompt.text) == ()
