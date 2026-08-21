import json
from pathlib import Path

import pytest

from research_dynamics.annotations import blinded_packet
from research_dynamics.contracts import (
    ConditionId,
    DeliberationPolicy,
    FrameworkKind,
    ProcessCondition,
    ProcessStudyConfig,
    VisibleMemoryPolicy,
)
from research_dynamics.extraction import extract_autoresearch_run
from research_dynamics.memory import render_memory_packet, select_memory_entries
from research_dynamics.metrics import summarize_annotations, summarize_decisions
from research_dynamics.orchestration import execute_manifest, plan_forks
from research_dynamics.openevolve_integration import instrument_openevolve_prompts
from research_dynamics.prompts import deliberation_block
from research_dynamics.protocol import ProcessProtocol


def _config(condition_id: ConditionId) -> ProcessStudyConfig:
    condition = ProcessCondition.for_id(condition_id)
    return ProcessStudyConfig(
        study_id="test-study",
        run_id=f"test-run-{condition_id.value}",
        framework=FrameworkKind.AUTORESEARCH,
        condition=condition,
        challenge_opportunities=(1, 3)
        if condition.deliberation_policy is DeliberationPolicy.ASSUMPTION_CHALLENGE
        else (),
    )


def test_four_conditions_map_only_to_memory_and_deliberation() -> None:
    assert ProcessCondition.for_id("RD0") == ProcessCondition(
        ConditionId.RD0,
        VisibleMemoryPolicy.SEQUENTIAL,
        DeliberationPolicy.NEUTRAL_REVIEW,
    )
    assert ProcessCondition.for_id("RD3") == ProcessCondition(
        ConditionId.RD3,
        VisibleMemoryPolicy.PORTFOLIO,
        DeliberationPolicy.ASSUMPTION_CHALLENGE,
    )
    with pytest.raises(ValueError, match="frozen treatment"):
        ProcessCondition(
            ConditionId.RD0,
            VisibleMemoryPolicy.PORTFOLIO,
            DeliberationPolicy.NEUTRAL_REVIEW,
        )


def test_config_is_strict_and_round_trips() -> None:
    condition = ProcessCondition.for_id(ConditionId.RD3)
    config = ProcessStudyConfig(
        study_id="oe-test",
        run_id="oe-test-rd3",
        framework=FrameworkKind.OPENEVOLVE,
        condition=condition,
        challenge_opportunities=(1,),
    )
    assert ProcessStudyConfig.from_dict(config.to_dict()) == config
    mutated = config.to_dict()
    mutated["reward"] = "changed"
    with pytest.raises(ValueError, match="fields differ"):
        ProcessStudyConfig.from_dict(mutated)


def test_deliberation_blocks_have_equal_character_exposure() -> None:
    blocks = [
        deliberation_block(challenge_condition=False, challenge_active=False),
        deliberation_block(challenge_condition=True, challenge_active=False),
        deliberation_block(challenge_condition=True, challenge_active=True),
    ]
    assert {len(block) for block in blocks} == {1800}
    assert "sealed" not in " ".join(blocks).lower()


def test_visible_memory_uses_only_public_fields_and_fixed_slots() -> None:
    records = [
        {
            "candidate_id": "a",
            "mechanism_hypothesis": "local attention",
            "evaluation": {
                "opportunity_index": 1,
                "execution_ok": True,
                "public_accuracy": 0.2,
                "private_accuracy": 1.0,
            },
            "retention_decision": "accept",
        },
        {
            "candidate_id": "b",
            "mechanism_hypothesis": "routing alternative",
            "evaluation": {
                "opportunity_index": 2,
                "execution_ok": True,
                "public_accuracy": 0.1,
                "sealed_score": 0.9,
            },
            "retention_decision": "reject",
        },
    ]
    sequential = select_memory_entries(records, VisibleMemoryPolicy.SEQUENTIAL)
    portfolio = select_memory_entries(records, VisibleMemoryPolicy.PORTFOLIO)
    assert len(sequential) == len(portfolio) == 4
    packet, _ = render_memory_packet(
        records, VisibleMemoryPolicy.PORTFOLIO, budget_chars=2000
    )
    assert len(packet) == 2000
    assert "private_accuracy" not in packet
    assert "sealed_score" not in packet


def test_fork_manifest_clones_one_checkpoint_into_all_cells(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.json"
    checkpoint.write_text("{}\n", encoding="utf-8")
    manifest_path = plan_forks(
        study_id="fork-test",
        framework=FrameworkKind.AUTORESEARCH,
        checkpoint=checkpoint,
        output_dir=tmp_path / "forks",
        command=["runner", "--output", "{output_dir}", "--seed", "{seed}"],
        horizon=4,
        seed=11,
        scientific=False,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {branch["condition_id"] for branch in manifest["branches"]} == {
        "RD0",
        "RD1",
        "RD2",
        "RD3",
    }
    configs = [
        ProcessStudyConfig.from_dict(
            json.loads(Path(branch["config_path"]).read_text(encoding="utf-8"))
        )
        for branch in manifest["branches"]
    ]
    assert len({config.source_checkpoint_hash for config in configs}) == 1
    dry = execute_manifest(manifest_path, dry_run=True)
    assert len(dry) == 4
    assert all(item["dry_run"] for item in dry)
    first_config = Path(manifest["branches"][0]["config_path"])
    tampered = json.loads(first_config.read_text(encoding="utf-8"))
    tampered["scientific"] = True
    first_config.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after randomization"):
        execute_manifest(manifest_path, dry_run=True)


def test_extraction_links_next_public_interpretation_and_blinds_treatment(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    artifacts = run / "artifacts"
    process = run / "research_process"
    artifacts.mkdir(parents=True)
    process.mkdir()
    records = []
    exposures = []
    for opportunity in (1, 2):
        candidate = {
            "metadata": {
                "research_current_explanation": f"explanation {opportunity}",
                "research_previous_interpretation": (
                    "not_applicable" if opportunity == 1 else "result contradicted claim"
                ),
                "research_previous_changed": (
                    "not_applicable" if opportunity == 1 else "yes"
                ),
            }
        }
        (artifacts / f"{opportunity:04d}.response.txt").write_text(
            json.dumps(candidate), encoding="utf-8"
        )
        records.append(
            {
                "candidate_id": f"c{opportunity}",
                "parent_id": "seed",
                "evaluation": {
                    "proposal_opportunity": opportunity,
                    "public_accuracy": opportunity / 10,
                },
                "retention_decision": "accept",
            }
        )
        exposures.append(
            {
                "opportunity": opportunity,
                "challenge_active": opportunity == 1,
                "memory_entries": [],
            }
        )
    (run / "lineage.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    (process / "exposures.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in exposures), encoding="utf-8"
    )
    decisions = extract_autoresearch_run(run, _config(ConditionId.RD1))
    assert decisions[0]["lab_note_after"]["changed_explanation"] == "yes"
    blinded = blinded_packet(decisions[0])
    encoded = json.dumps(blinded)
    assert "RD1" not in encoded
    assert "autoresearch" not in encoded
    assert "condition_id" not in encoded


def test_process_summary_keeps_primary_outcomes_process_based() -> None:
    result = summarize_annotations(
        [
            {
                "research_move": "ablation",
                "epistemic_purpose": "distinguish_explanations",
                "evidence_response": "reject",
                "research_displacement": "D4_assumption_or_problem_change",
                "discriminating_experiment": True,
                "prediction_contradicted": True,
                "rationale_action_aligned": True,
                "interpretation_supported_by_result": True,
                "hypothesis_id": "h1",
            },
            {
                "research_move": "alternative_mechanism",
                "epistemic_purpose": "falsify",
                "evidence_response": "retain",
                "research_displacement": "D3_alternative_mechanism",
                "discriminating_experiment": False,
                "prediction_contradicted": False,
                "rationale_action_aligned": True,
                "interpretation_supported_by_result": False,
                "hypothesis_id": "h2",
            },
        ]
    )
    assert result["primary"]["discriminating_experiment_rate"] == 0.5
    assert result["primary"][
        "evidence_responsive_revision_rate_given_contradiction"
    ] == 1.0
    assert "benchmark" not in result["primary"]


def test_openevolve_prompt_instrumentation_adds_dynamic_treatment_in_worker_shape(
    tmp_path: Path,
) -> None:
    from openevolve.config import load_config
    from openevolve.prompt.sampler import PromptSampler

    condition = ProcessCondition.for_id(ConditionId.RD3)
    config = ProcessStudyConfig(
        study_id="oe-worker-test",
        run_id="oe-worker-test-rd3",
        framework=FrameworkKind.OPENEVOLVE,
        condition=condition,
        challenge_opportunities=(1,),
    )
    protocol = ProcessProtocol(config, tmp_path / "run")
    vendor_config = load_config(
        Path(__file__).resolve().parents[1]
        / "agents"
        / "openevolve_generic"
        / "config.yaml"
    )
    with instrument_openevolve_prompts(protocol):
        sampler = PromptSampler(vendor_config.prompt)
        prompt = sampler.build_prompt(
            current_program="{}",
            parent_program="{}",
            program_metrics={"public_accuracy": 0.1},
            previous_programs=[],
            top_programs=[],
            inspirations=[],
            language="json",
            evolution_round=1,
            diff_based_evolution=False,
        )
    assert "assumption challenge" in prompt["user"].lower()
    assert "research_current_explanation" in prompt["user"]
    exposures = (tmp_path / "run" / "research_process" / "exposures.jsonl")
    assert json.loads(exposures.read_text(encoding="utf-8"))["opportunity"] == 1


def test_annotation_free_summary_reports_idea_and_lineage_dynamics() -> None:
    result = summarize_decisions(
        [
            {
                "opportunity": 1,
                "candidate_id": "c1",
                "parent_id": "seed",
                "challenge_active": True,
                "memory_entry_ids": ["old-1"],
                "lab_note_before": {
                    "research_current_explanation": "attention fixes carry propagation",
                    "research_next_experiment": "ablate attention path",
                    "research_evidence": "old-1 failed publicly",
                    "research_challenged_assumption": "attention is necessary",
                },
                "lab_note_after": {"changed_explanation": "yes"},
            },
            {
                "opportunity": 2,
                "candidate_id": "c2",
                "parent_id": "seed",
                "challenge_active": False,
                "memory_entry_ids": [],
                "lab_note_before": {
                    "research_current_explanation": "routing handles carry state",
                    "research_next_experiment": "test fixed routing",
                },
                "lab_note_after": {"changed_explanation": "no"},
            },
        ]
    )
    assert result["idea_and_logic_diversity"]["unique_explanation_rate"] == 1.0
    assert result["lineage"]["maximum_children_from_one_parent"] == 2
    assert result["intervention_process"]["challenge_uptake_rate"] == 1.0
