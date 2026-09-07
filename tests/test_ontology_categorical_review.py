import json
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.ontology_categorical_fingerprint import (
    CategoricalFingerprintError,
    SchemaExtensionRequired,
    campaign_schema,
    family_id,
    load_contract,
    output_document,
    schema_revision,
    validate_fingerprint,
)
from experiments.ontology_categorical_review import (
    REVIEW_VERSION,
    canonicalize_observations,
    source_digest,
    source_inventory,
    validate_review_schema,
    validate_role_consistency,
    validate_source_review,
)


def component(schema, name):
    return next(c for c in schema["components"] if c["id"] == name)


def empty_fingerprint(schema):
    return {c["id"]: "absent" for c in schema["components"]}


def activation_review():
    """A small source-reviewed fixture, not a real campaign annotation."""
    schema = campaign_schema("fashion")
    sources = {"model.py": "def forward(x):\n    return F.silu(x)\n"}
    statements = source_inventory(sources)
    returned = next(s["id"] for s in statements if s["kind"] == "Return")
    fingerprint = dict(empty_fingerprint(schema), activation="silu")
    review = {
        "version": REVIEW_VERSION,
        "reviewer": "synthetic regression fixture",
        "source_sha256": source_digest(sources),
        "schema_revision": schema_revision(schema),
        "entrypoint_analysis": "forward directly returns the feature nonlinearity.",
        "facts": [
            {
                "id": "nonlinearity",
                "component": "activation",
                "aspect": "feature nonlinear map",
                "statements": [returned],
                "observations": ["F.silu"],
                "role_reason": "The function transforms the feature stream directly.",
            }
        ],
        "components": {
            c["id"]: {
                "facts": [],
                "reason": "No such role exists in this isolated function.",
            }
            for c in schema["components"]
        },
        "statement_coverage": {
            s["id"]: {
                "kind": "mechanism" if s["id"] == returned else "covered_by_composite",
                "facts": ["nonlinearity"],
                "reason": "Feature transform or its enclosing declaration.",
            }
            for s in statements
        },
    }
    review["components"]["activation"] = {
        "facts": ["nonlinearity"],
        "reason": "Direct SiLU transform of the input.",
    }
    return schema, sources, fingerprint, review


def test_all_components_have_auditable_role_boundaries_and_unambiguous_aliases():
    for schema in load_contract()["campaigns"]:
        validate_review_schema(schema)
        for c in schema["components"]:
            for alias, canonical in c["aliases"].items():
                assert canonicalize_observations(c, [alias, canonical]) == canonical
            for value in c["values"]:
                assert canonicalize_observations(c, [value, value.upper()]) == value
                assert canonicalize_observations(c, [value] * 3) == value


@pytest.mark.parametrize(
    "campaign, role, equivalent",
    [
        ("fashion", "activation", ["SiLU", "silu", "nn.SiLU", "F.silu"]),
        ("kws", "normalization", ["LayerNorm", "nn.LayerNorm", "F.layer_norm"]),
        ("nanogpt", "normalization", ["RMSNorm", "nn.RMSNorm", "F.rms_norm"]),
        ("kws", "state_update", ["nn.GRU", "nn.GRUCell"]),
        ("fashion", "view_weight_policy", ["uniform", "fixed_shared", "view_specific"]),
        (
            "fashion",
            "view_aggregation",
            ["probability_mixture", "log_probability_mixture"],
        ),
    ],
)
def test_equivalent_role_implementations_do_not_create_new_families(
    campaign, role, equivalent
):
    schema = campaign_schema(campaign)
    definition = component(schema, role)
    vectors = []
    for alias in equivalent:
        vector = empty_fingerprint(schema)
        vector[role] = canonicalize_observations(definition, [alias])
        vectors.append(vector)
    assert len({family_id(vector, schema) for vector in vectors}) == 1


@pytest.mark.parametrize(
    "role, values",
    [
        ("activation", ["relu", "silu"]),
        ("normalization", ["layer_normalization", "rms_normalization"]),
        ("attention_weighting", ["softmax", "unnormalized_weights"]),
        ("optimization_rule", ["adam", "adamw"]),
        ("output_link", ["softmax", "sigmoid"]),
        ("token_embedding", ["learned_lookup", "factorized_lookup"]),
    ],
)
def test_distinct_mechanisms_are_not_merged_or_concatenated(role, values):
    schema = campaign_schema("nanogpt")
    definition = component(schema, role)
    with pytest.raises(SchemaExtensionRequired, match="split roles"):
        canonicalize_observations(definition, values)
    fingerprints = [
        dict(empty_fingerprint(schema), **{role: value}) for value in values
    ]
    assert family_id(fingerprints[0], schema) != family_id(fingerprints[1], schema)


@pytest.mark.parametrize(
    "role, label",
    [
        ("activation", "SiLU + silu"),
        ("normalization", "LayerNorm + rms_norm"),
        ("output_link", "sigmoid + softmax"),
        ("token_embedding", "lookup_rank_four"),
    ],
)
def test_raw_bags_of_symbols_and_numeric_proxy_labels_fail(role, label):
    schema = campaign_schema("nanogpt")
    with pytest.raises(SchemaExtensionRequired):
        canonicalize_observations(component(schema, role), [label])


def test_gate_and_output_roles_are_independent_of_main_activation():
    schema = campaign_schema("fashion")
    base = dict(empty_fingerprint(schema), activation="silu")
    gate = dict(
        base,
        feature_modulation="multiplicative_gate",
        modulation_source="global_channel_summary",
        modulation_transfer="sigmoid",
    )
    probability_output = dict(base, output_link="sigmoid")
    for fp in (base, gate, probability_output):
        validate_role_consistency(fp, schema)
    assert len({family_id(fp, schema) for fp in (base, gate, probability_output)}) == 3
    assert gate["activation"] == probability_output["activation"] == base["activation"]
    with pytest.raises(CategoricalFingerprintError, match="Gate detail"):
        validate_role_consistency(dict(base, modulation_transfer="sigmoid"), schema)


def test_alias_must_not_cross_component_roles():
    schema = campaign_schema("fashion")
    with pytest.raises(SchemaExtensionRequired):
        canonicalize_observations(
            component(schema, "activation"), ["global_channel_summary"]
        )
    with pytest.raises(SchemaExtensionRequired):
        canonicalize_observations(component(schema, "modulation_source"), ["sigmoid"])


def test_new_schema_values_need_definitions_and_aliases_cannot_conflict():
    schema = campaign_schema("fashion")
    component(schema, "activation")["values"].append("new_nonlinearity")
    with pytest.raises(CategoricalFingerprintError, match="exact definition"):
        validate_review_schema(schema)
    schema = campaign_schema("fashion")
    component(schema, "activation")["aliases"]["SILU"] = "relu"
    with pytest.raises(CategoricalFingerprintError, match="conflicting alias"):
        validate_review_schema(schema)


def test_broad_draft_categories_cannot_be_published_as_complete():
    schema = campaign_schema("fashion")
    fp = dict(empty_fingerprint(schema), optimization_rule="adaptive_gradient")
    validate_fingerprint(fp, schema)
    with pytest.raises(SchemaExtensionRequired, match="requires refinement"):
        validate_role_consistency(fp, schema)


def test_source_review_is_bound_to_current_code_schema_and_full_coverage():
    schema, sources, fp, review = activation_review()
    assert validate_source_review(review, sources, fp, schema)["fact_count"] == 1
    with pytest.raises(CategoricalFingerprintError, match="digest"):
        validate_source_review(review, {"model.py": "return_value = 3\n"}, fp, schema)
    stale = dict(review, schema_revision="old")
    with pytest.raises(CategoricalFingerprintError, match="revision"):
        validate_source_review(stale, sources, fp, schema)
    missing = deepcopy(review)
    missing["statement_coverage"].pop(next(iter(missing["statement_coverage"])))
    with pytest.raises(CategoricalFingerprintError, match="coverage"):
        validate_source_review(missing, sources, fp, schema)


def test_omitted_mechanisms_and_duplicate_ownership_are_rejected():
    schema, sources, fp, review = activation_review()
    missing = deepcopy(review)
    missing["components"]["activation"]["facts"] = []
    with pytest.raises(CategoricalFingerprintError, match="no source evidence"):
        validate_source_review(missing, sources, fp, schema)
    duplicate = deepcopy(review)
    duplicate["facts"].append(dict(duplicate["facts"][0], id="same_fact_again"))
    with pytest.raises(CategoricalFingerprintError, match="Duplicate ownership"):
        validate_source_review(duplicate, sources, fp, schema)
    excluded = deepcopy(review)
    statement = review["facts"][0]["statements"][0]
    excluded["statement_coverage"][statement] = {
        "kind": "unreachable",
        "reason": "This conflicts with its active fact.",
    }
    with pytest.raises(CategoricalFingerprintError, match="excluded"):
        validate_source_review(excluded, sources, fp, schema)


def test_legacy_complete_flag_is_not_a_source_review():
    schema, sources, fp, _ = activation_review()
    with pytest.raises(CategoricalFingerprintError, match="role_evidence"):
        validate_source_review({"fingerprint_complete": True}, sources, fp, schema)


def test_publication_reviews_shared_seed_once_and_refuses_ungated_output():
    _schema, sources, fp, review = activation_review()
    seed_row = {
        "proposal": 0,
        "candidate_id": "seed",
        "ontology_parent_id": None,
        "is_seed": True,
        "canonical_seed_id": "seed",
    }
    runs = {"first": [seed_row], "second": [seed_row]}
    with pytest.raises(CategoricalFingerprintError, match="requires source_reviews"):
        output_document("fashion", runs, canonical_seed_fingerprints={"seed": fp})
    result = output_document(
        "fashion",
        runs,
        canonical_seed_fingerprints={"seed": fp},
        source_reviews={"seed": review},
        source_bundles={"seed": sources},
    )
    assert len(result["source_reviews"]) == len(result["review_audits"]) == 1
    assert result["runs"]["first"][0]["fingerprint"] == fp
    assert result["runs"]["second"][0]["fingerprint"] == fp
    assert all(v == 0 for v in result["runs"]["second"][0]["metrics"].values())
    assert "fingerprint" not in seed_row  # Input references are not mutated.


def test_conflicting_run_seed_cannot_override_the_shared_source_review():
    schema, sources, fp, review = activation_review()
    row = {
        "proposal": 0,
        "candidate_id": "seed",
        "ontology_parent_id": None,
        "is_seed": True,
        "canonical_seed_id": "seed",
        "fingerprint": empty_fingerprint(schema),
    }
    with pytest.raises(CategoricalFingerprintError, match="disagrees"):
        output_document(
            "fashion",
            {"run": [row]},
            canonical_seed_fingerprints={"seed": fp},
            source_reviews={"seed": review},
            source_bundles={"seed": sources},
        )


def test_fork_working_schema_is_embedded_without_mutating_shared_registry():
    schema, sources, fp, review = activation_review()
    working = deepcopy(schema)
    component(working, "activation")["scope"] += " The fork clarified this role."
    local_review = dict(review, schema_revision=schema_revision(working))
    runs = {
        "run": [
            {
                "proposal": 0,
                "candidate_id": "seed",
                "ontology_parent_id": None,
                "fingerprint": fp,
            }
        ]
    }
    result = output_document(
        "fashion",
        runs,
        schema_override=working,
        source_reviews={"seed": local_review},
        source_bundles={"seed": sources},
    )
    assert result["schema"] == working
    assert result["schema_revision"] != schema_revision(schema)
    assert campaign_schema("fashion") == schema
    with pytest.raises(CategoricalFingerprintError, match="revision"):
        output_document(
            "fashion",
            runs,
            schema_override=working,
            source_reviews={"seed": review},
            source_bundles={"seed": sources},
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("id", "har"),
        ("included_conditions", ["C4"]),
        ("proposal_limit", 999),
    ],
)
def test_fork_schema_cannot_expand_study_scope(field, value):
    working = campaign_schema("fashion")
    working[field] = value
    with pytest.raises(CategoricalFingerprintError, match="cannot change study scope"):
        output_document("fashion", {}, schema_override=working)


def test_unused_function_is_accounted_for_without_changing_the_fingerprint():
    schema, sources, fp, review = activation_review()
    baseline_family = family_id(fp, schema)
    # The entrypoint does not call this helper; its ReLU must not be collected
    # into the active activation component merely because it appears in source.
    sources["unused.py"] = "def unused(x):\n    return F.relu(x)\n"
    review["source_sha256"] = source_digest(sources)
    for statement in source_inventory(sources):
        if statement["file"] == "unused.py":
            review["statement_coverage"][statement["id"]] = {
                "kind": "unreachable",
                "reason": "No import or call from the isolated forward entrypoint.",
            }
    validate_source_review(review, sources, fp, schema)
    assert family_id(fp, schema) == baseline_family


def test_actual_channel_recalibration_source_owns_gate_roles_not_backbone_activation():
    # The side-chat counterexample is used as a regression fixture only. Its
    # historical C4 run is not imported into the active C0-C3 diagnostic.
    root = Path(__file__).resolve().parents[1]
    references = json.loads(
        (root / "experiments/ontology_fashion_candidate_references_v2.json").read_text(
            encoding="utf-8"
        )
    )
    gate_source = references["reviewed_profiles"][0]["program_parts"][
        "ChannelRecalibration"
    ]
    schema = campaign_schema("fashion")
    sources = {"gate.py": gate_source}
    statements = source_inventory(sources)
    projection = next(
        s["id"]
        for s in statements
        if s["kind"] == "Assign" and "self.projection =" in s["code"]
    )
    scale = next(
        s["id"] for s in statements if s["kind"] == "Assign" and "scale =" in s["code"]
    )
    multiply = next(s["id"] for s in statements if s["kind"] == "Return")
    fp = dict(
        empty_fingerprint(schema),
        feature_modulation="multiplicative_gate",
        modulation_source="global_channel_summary",
        modulation_transfer="sigmoid",
        gate_hidden_activation="silu",
    )
    specifications = [
        (
            "feature_modulation",
            "feature multiplication",
            multiply,
            ["multiplicative_gate"],
        ),
        (
            "modulation_source",
            "control information source",
            projection,
            ["global_channel_summary"],
        ),
        ("modulation_transfer", "control transfer function", scale, ["torch.sigmoid"]),
        (
            "gate_hidden_activation",
            "control hidden nonlinearity",
            projection,
            ["nn.SiLU"],
        ),
    ]
    facts = [
        {
            "id": role,
            "component": role,
            "aspect": aspect,
            "statements": [statement],
            "observations": observations,
            "role_reason": "Pooled channel-control projection scales features.",
        }
        for role, aspect, statement, observations in specifications
    ]
    coverage = {}
    for statement in statements:
        owners = [f["id"] for f in facts if statement["id"] in f["statements"]]
        coverage[statement["id"]] = {
            "kind": "mechanism" if owners else "covered_by_composite",
            "facts": owners or ["feature_modulation"],
            "reason": "Gate computation or its module construction and initialization.",
        }
    review = {
        "version": REVIEW_VERSION,
        "reviewer": "retained source regression fixture",
        "schema_revision": schema_revision(schema),
        "source_sha256": source_digest(sources),
        "entrypoint_analysis": "The forward projection controls the feature gate.",
        "facts": facts,
        "statement_coverage": coverage,
        "components": {
            c["id"]: {
                "facts": [c["id"]] if c["id"] in {f["id"] for f in facts} else [],
                "reason": "This isolated module only adds a channel-control gate.",
            }
            for c in schema["components"]
        },
    }
    validate_source_review(review, sources, fp, schema)
    assert fp["activation"] == "absent"
    assert fp["output_link"] == "absent"
    # Changing a scale constant preserves the reviewed categorical mechanism.
    scaled_sources = {
        "gate.py": gate_source.replace("2.0 * torch.sigmoid", "3.0 * torch.sigmoid")
    }
    scaled_review = dict(review, source_sha256=source_digest(scaled_sources))
    validate_source_review(scaled_review, scaled_sources, fp, schema)
    assert family_id(fp, schema) == family_id(dict(fp), schema)
