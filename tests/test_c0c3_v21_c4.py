from __future__ import annotations

import json
from pathlib import Path

from experiments import c0c3_v21_c4_guard as guard
from experiments.c0c3_factorial import state as state_module
from experiments.c0c3_factorial.periodic_refresh import phase_search_seed
from experiments.c0c3_factorial.prompts import PromptContext, VisibleCandidate
from experiments.c0c3_factorial.spec import (
    C4_CONDITION,
    OPENEVOLVE_V2_EXECUTION_RULE,
    BudgetSpec,
    ConversationMode,
    FactorialSpec,
    ModelSpec,
    conditions_for_protocol,
    make_assignments,
)
from experiments.c0c3_factorial.state import (
    Candidate,
    Evaluation,
    SearchController,
    Usage,
)


def _spec(*, blocks: int = 2) -> FactorialSpec:
    return FactorialSpec(
        protocol_version="2.1",
        study_id="c4-test",
        study_seed=123,
        blocks=blocks,
        portfolio_capacity=4,
        transition_opportunities=tuple(range(10, 201, 10)),
        model=ModelSpec("gpt-test", "high"),
        budget=BudgetSpec(
            proposals=200,
            candidate_evaluations=200,
            max_total_tokens=100_000_000,
            max_evaluator_seconds=10_000.0,
            evaluator_timeout_seconds=100,
        ),
        conversation_mode=ConversationMode.EPHEMERAL,
        execution_rule=OPENEVOLVE_V2_EXECUTION_RULE,
        include_no_search=False,
    )


def _seed() -> Candidate:
    return Candidate(
        candidate_id="seed",
        parent_ids=[],
        fitness=1.0,
        metrics={"score": 1.0},
        artifact_path="candidates/seed",
        hypothesis="starting design",
        intended_edit="none",
        created_opportunity=0,
        retained_order=0,
    )


def test_v21_assignments_add_c4_without_changing_legacy_enum_iteration() -> None:
    spec = _spec()
    assignments = make_assignments(spec, task_id="task", framework_id="openevolve")
    assert [condition.value for condition in conditions_for_protocol("2.1")] == [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
    ]
    for block in (1, 2):
        rows = [row for row in assignments if row.block == block]
        assert {row.condition.value for row in rows} == {
            "C0",
            "C1",
            "C2",
            "C3",
            "C4",
        }
        assert len({row.run_seed for row in rows}) == 1
    assert not C4_CONDITION.has_portfolio
    assert not C4_CONDITION.transition_active(10, (10,))
    assert len(conditions_for_protocol("1.7")) == 4


def test_c4_refresh_keeps_private_accounting_but_resets_visible_epoch(
    tmp_path: Path,
) -> None:
    spec = _spec(blocks=1)
    run_dir = tmp_path / "run"
    controller = SearchController.create(
        run_dir,
        spec,
        run_id="b01-c4",
        condition=C4_CONDITION,
        seed_candidate=_seed(),
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "assignment": {
                    "block": 1,
                    "condition": "C4",
                    "run_id": "b01-c4",
                    "run_seed": 42,
                },
                "periodic_full_refresh": {"interval_proposals": 10},
            }
        ),
        encoding="utf-8",
    )
    for opportunity in range(1, 11):
        controller.begin()
        controller.complete(
            candidate_id=f"failed-{opportunity}",
            artifact_path=f"candidates/failed-{opportunity}",
            hypothesis="failed idea",
            intended_edit="failed edit",
            evaluation=Evaluation(
                valid=False,
                fitness=None,
                metrics={},
                evaluator_seconds=1.0,
                evaluator_calls=1,
                failure_kind="nonqualification",
            ),
            usage=Usage(input_tokens=10, output_tokens=2),
            prompt_hashes={},
        )

    guard._refresh_if_due(run_dir, spec=spec, state_module=state_module)
    refreshed = SearchController.load(run_dir, spec)
    memory = json.loads((run_dir / "subject-memory.json").read_text())
    assert refreshed.state.proposals_used == 10
    assert refreshed.state.evaluations_used == 10
    assert refreshed.state.usage.total_tokens == 120
    assert refreshed.state.next_opportunity == 11
    assert refreshed.state.condition == "C4"
    assert list(refreshed.state.candidates) == ["seed"]
    assert memory["history_start_opportunity"] == 11
    assert memory["phase"] == 2
    assert guard._epoch_accounting(run_dir, minimum=11) == {
        "evaluations": 0.0,
        "tokens": 0.0,
        "evaluator_seconds": 0.0,
    }
    assert sum(
        json.loads(line).get("event") == "proposal_completed"
        for line in (run_dir / "events.jsonl").read_text().splitlines()
    ) == 10

    visible = guard._fresh_prompt_context(
        PromptContext(
            condition=C4_CONDITION,
            opportunity=11,
            selected_parent_id="seed",
            visible_candidates=(
                VisibleCandidate("seed", 1.0, {"score": 1.0}, 0, "visible/slot-1"),
            ),
            remaining_proposals=190,
            remaining_evaluations=190,
            remaining_tokens=99_999_880,
            remaining_evaluator_seconds=9_990.0,
        ),
        spec=spec,
        run_dir=run_dir,
    )
    assert visible.opportunity == 1
    assert visible.remaining_proposals == 200
    assert visible.remaining_evaluations == 200
    assert visible.remaining_tokens == 100_000_000
    assert visible.remaining_evaluator_seconds == 10_000.0
    refreshed.begin()
    assert guard._physical_active_opportunity(run_dir) == 11


def test_c4_search_seed_has_an_independent_namespace() -> None:
    semantic = phase_search_seed(42, 11)
    c4 = phase_search_seed(
        42,
        11,
        namespace="greedy-openevolve-v2.1-c4-refresh",
    )
    assert semantic != c4
