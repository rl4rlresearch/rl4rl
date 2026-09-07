from __future__ import annotations

import json
from copy import deepcopy

import pytest

from experiments.ontology_categorical_fingerprint import (
    CategoricalFingerprintError,
    SchemaExtensionRequired,
    annotate_run,
    campaign_schema,
    family_id,
    load_contract,
    schema_revision,
    validate_fingerprint,
)
from experiments.ontology_categorical_inventory import build_campaign_inventory


def addition_fingerprint(**updates):
    result = {
        component["id"]: "absent"
        for component in campaign_schema("addition")["components"]
    }
    result.update(updates)
    return result


def test_family_is_exact_complete_categorical_vector():
    schema = campaign_schema("addition")
    first = addition_fingerprint(
        operand_representation="paired_decimal_digits", token_embedding="learned_lookup"
    )
    same = dict(first)
    different = dict(first, token_embedding="factorized_lookup")
    assert validate_fingerprint(first, schema) == first
    assert family_id(first, schema) == family_id(same, schema)
    assert family_id(first, schema) != family_id(different, schema)


def test_all_categorical_campaigns_are_in_scope():
    contract = load_contract()
    assert {campaign["id"] for campaign in contract["campaigns"]} == {
        "addition",
        "nanogpt",
        "fashion",
        "kws",
        "har",
        "tiny_adderboard",
    }
    har = campaign_schema("har")
    assert har["source_campaign_key"] == "uci_har_pareto_v21"
    assert har["data_directory"] == "uci-har-pareto-v21"
    assert har["included_conditions"] == ["C0", "C1", "C2", "C3"]
    assert har.get("proposal_limit") is None
    tiny = campaign_schema("tiny_adderboard")
    assert tiny["source_campaign_key"] == "tiny_adderboard_v21"
    assert tiny["data_directory"] == "tiny-v21"
    assert tiny["included_conditions"] == ["C0", "C1", "C2", "C3"]
    assert tiny["proposal_limit"] == 100
    assert campaign_schema("addition")["proposal_limit"] == 120
    assert contract["excluded_campaigns"] == []


def test_fingerprint_rejects_numeric_and_undeclared_categories():
    schema = campaign_schema("addition")
    with pytest.raises(CategoricalFingerprintError):
        validate_fingerprint(addition_fingerprint(token_embedding="lookup_two"), schema)
    with pytest.raises(SchemaExtensionRequired):
        validate_fingerprint(
            addition_fingerprint(token_embedding="symbolic_embedding"), schema
        )
    with pytest.raises(CategoricalFingerprintError):
        validate_fingerprint(addition_fingerprint(token_embedding=3), schema)


def test_living_schema_extension_changes_revision_without_creating_other_bin():
    schema = campaign_schema("addition")
    before = schema_revision(schema)
    extended = deepcopy(schema)
    component = next(
        item for item in extended["components"] if item["id"] == "token_embedding"
    )
    component["values"].append("symbolic_embedding")
    candidate = addition_fingerprint(token_embedding="symbolic_embedding")
    assert (
        validate_fingerprint(candidate, extended)["token_embedding"]
        == "symbolic_embedding"
    )
    assert schema_revision(extended) != before
    assert load_contract()["campaign_schema_policy"]["status"] == "living"


def test_new_component_requires_absent_reprojection_of_earlier_candidates():
    schema = campaign_schema("addition")
    extended = deepcopy(schema)
    extended["components"].append(
        {
            "id": "auxiliary_memory",
            "description": "A newly discovered memory mechanism.",
            "values": ["none", "retrieval_memory", "absent"],
        }
    )
    prior = addition_fingerprint()
    with pytest.raises(CategoricalFingerprintError, match="missing"):
        validate_fingerprint(prior, extended)
    reprojection = dict(prior, auxiliary_memory="absent")
    assert validate_fingerprint(reprojection, extended) == reprojection


@pytest.mark.parametrize("return_id", ["reversion", "seed"])
def test_reversion_counts_as_an_edit_but_not_new_component_state_or_family(return_id):
    schema = campaign_schema("addition")
    seed = addition_fingerprint(
        operand_representation="paired_decimal_digits", token_embedding="learned_lookup"
    )
    first = dict(seed, token_embedding="factorized_lookup")
    reversion = dict(first, token_embedding="learned_lookup")
    rows = annotate_run(
        [
            {
                "proposal": 0,
                "candidate_id": "seed",
                "ontology_parent_id": None,
                "fingerprint": seed,
            },
            {
                "proposal": 1,
                "candidate_id": "first",
                "ontology_parent_id": "seed",
                "fingerprint": first,
            },
            {
                "proposal": 2,
                "candidate_id": return_id,
                "ontology_parent_id": "first",
                "fingerprint": reversion,
            },
        ],
        schema,
    )
    assert rows[1]["metrics"] == {
        "component_edits_marginal": 1,
        "component_edits_cumulative": 1,
        "new_component_states_marginal": 1,
        "new_component_states_cumulative": 1,
        "family_switches_marginal": 1,
        "family_switches_cumulative": 1,
        "new_families_marginal": 1,
        "new_families_cumulative": 1,
    }
    assert rows[2]["changed_components"] == ["token_embedding"]
    assert rows[2]["metrics"] == {
        "component_edits_marginal": 1,
        "component_edits_cumulative": 2,
        "new_component_states_marginal": 0,
        "new_component_states_cumulative": 1,
        "family_switches_marginal": 1,
        "family_switches_cumulative": 2,
        "new_families_marginal": 0,
        "new_families_cumulative": 1,
    }


def test_new_family_can_use_previously_seen_component_states():
    schema = campaign_schema("addition")
    seed = addition_fingerprint(
        operand_representation="paired_decimal_digits", token_embedding="learned_lookup"
    )
    first = dict(
        seed,
        operand_representation="one_hot_operand_tokens",
        token_embedding="factorized_lookup",
    )
    novel_combination = dict(first, operand_representation="paired_decimal_digits")
    rows = annotate_run(
        [
            {
                "proposal": 0,
                "candidate_id": "seed",
                "ontology_parent_id": None,
                "fingerprint": seed,
            },
            {
                "proposal": 1,
                "candidate_id": "first",
                "ontology_parent_id": "seed",
                "fingerprint": first,
            },
            {
                "proposal": 2,
                "candidate_id": "combination",
                "ontology_parent_id": "first",
                "fingerprint": novel_combination,
            },
        ],
        schema,
    )
    assert rows[2]["metrics"]["component_edits_marginal"] == 1
    assert rows[2]["metrics"]["new_component_states_marginal"] == 0
    assert rows[2]["metrics"]["new_families_marginal"] == 1


def test_primary_parent_must_be_earlier_in_the_run():
    schema = campaign_schema("addition")
    fingerprint = addition_fingerprint()
    with pytest.raises(CategoricalFingerprintError, match="occur earlier"):
        annotate_run(
            [
                {
                    "proposal": 0,
                    "candidate_id": "seed",
                    "ontology_parent_id": None,
                    "fingerprint": fingerprint,
                },
                {
                    "proposal": 1,
                    "candidate_id": "child",
                    "ontology_parent_id": "later",
                    "fingerprint": fingerprint,
                },
            ],
            schema,
        )


def test_reused_candidate_must_keep_its_fingerprint_and_each_proposal_is_unique():
    schema = campaign_schema("addition")
    fp = addition_fingerprint(token_embedding="learned_lookup")
    seed = {
        "proposal": 0,
        "candidate_id": "seed",
        "ontology_parent_id": None,
        "fingerprint": fp,
    }
    conflicting = {
        "proposal": 1,
        "candidate_id": "seed",
        "ontology_parent_id": "seed",
        "fingerprint": dict(fp, token_embedding="factorized_lookup"),
    }
    with pytest.raises(CategoricalFingerprintError, match="inconsistent fingerprints"):
        annotate_run([seed, conflicting], schema)
    repeated_proposal = dict(seed, candidate_id="another")
    with pytest.raises(CategoricalFingerprintError, match="Duplicate proposal"):
        annotate_run([seed, repeated_proposal], schema)


@pytest.mark.parametrize(
    ("campaign_id", "proposal_limit"),
    [("addition", 120), ("tiny_adderboard", 100)],
)
def test_campaign_inventory_caps_proposals_and_excludes_nothing_else(
    tmp_path, campaign_id, proposal_limit
):
    schema = campaign_schema(campaign_id)
    root = tmp_path / schema["data_directory"]
    run_id = f"{campaign_id}-run"
    run = root / "runs" / run_id
    excluded_run_id = f"{campaign_id}-c-four-run"
    excluded_run = root / "runs" / excluded_run_id
    (run / "candidates" / "seed" / "src").mkdir(parents=True)
    (run / "candidates" / "one" / "src").mkdir(parents=True)
    (run / "candidates" / "boundary" / "src").mkdir(parents=True)
    (run / "candidates" / "late" / "src").mkdir(parents=True)
    for candidate in ("seed", "one", "boundary", "late"):
        (run / "candidates" / candidate / "src" / "model.py").write_text(
            "x = 1\n", encoding="utf-8"
        )
    (excluded_run / "candidates" / "seed" / "src").mkdir(parents=True)
    (excluded_run / "candidates" / "seed" / "src" / "model.py").write_text(
        "x = 1\n", encoding="utf-8"
    )
    (excluded_run / "manifest.json").write_text(
        json.dumps(
            {"baseline": {"candidate_id": "seed"}, "assignment": {"condition": "C4"}}
        ),
        encoding="utf-8",
    )
    (excluded_run / "events.jsonl").write_text("", encoding="utf-8")
    root.mkdir(exist_ok=True)
    (root / "campaign.json").write_text(
        json.dumps({"primary_run_ids": [run_id, excluded_run_id]}), encoding="utf-8"
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {"baseline": {"candidate_id": "seed"}, "assignment": {"condition": "C0"}}
        ),
        encoding="utf-8",
    )
    (run / "events.jsonl").write_text(
        "\n".join(
            json.dumps(item)
            for item in [
                {
                    "event": "proposal_completed",
                    "opportunity": 1,
                    "candidate_id": "one",
                    "parent_ids": ["seed"],
                    "evaluation": {"valid": True},
                },
                {
                    "event": "proposal_completed",
                    "opportunity": proposal_limit,
                    "candidate_id": "boundary",
                    "parent_ids": ["one"],
                    "evaluation": {"valid": True},
                },
                {
                    "event": "proposal_completed",
                    "opportunity": proposal_limit + 1,
                    "candidate_id": "late",
                    "parent_ids": ["one"],
                    "evaluation": {"valid": True},
                },
            ]
        ),
        encoding="utf-8",
    )
    inventory = build_campaign_inventory(campaign_id, tmp_path)
    records = inventory["runs"][0]["records"]
    assert [record["proposal"] for record in records] == [0, 1, proposal_limit]
    assert records[1]["ontology_parent_id"] == "seed"
    assert records[0]["canonical_seed_id"] == "seed"
    assert inventory["canonical_seeds"] == [
        {
            "candidate_id": "seed",
            "classification_status": "pending_categorical_annotation",
            "source_status": "available",
            "source_path": "candidates/seed",
            "run_ids": [run_id],
        }
    ]
    assert not inventory["problems"]
    assert inventory["excluded_runs"] == [
        {
            "run_id": excluded_run_id,
            "condition": "C4",
            "reason": "condition outside categorical C0 C3 study scope",
        }
    ]
