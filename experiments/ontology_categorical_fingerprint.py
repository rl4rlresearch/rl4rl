"""Exact categorical ontology fingerprints and trajectory metrics.

This module deliberately does not reuse the legacy semantic ``family_signature``
or its preserving-edge graph.  A family is an equality class of the complete
categorical fingerprint defined by the current campaign-schema revision.

The common contract is stable.  Campaign schemas are living registries: a new
mechanism extends the registry, then every proposal in that campaign is
reprojected and all metrics are recomputed under the resulting revision.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "ontology-categorical-fingerprint-v1.json"
)
FINGERPRINT_VERSION = "categorical_fingerprint_v1"
ABSENT = "absent"
_CATEGORY = re.compile(r"^[a-z][a-z_]*$")

METRICS = {
    "component_edits_marginal": (
        "Number of categorical component values that differ from the primary parent."
    ),
    "component_edits_cumulative": "Running sum of component_edits_marginal.",
    "new_component_states_marginal": (
        "Changed component values not seen previously for that component in this run."
    ),
    "new_component_states_cumulative": "Running sum of new_component_states_marginal.",
    "family_switches_marginal": (
        "One when the complete fingerprint differs from the primary parent; "
        "zero otherwise."
    ),
    "family_switches_cumulative": "Running sum of family_switches_marginal.",
    "new_families_marginal": (
        "One when a switched complete fingerprint is absent from earlier run history; "
        "zero otherwise."
    ),
    "new_families_cumulative": "Running sum of new_families_marginal.",
}


class CategoricalFingerprintError(ValueError):
    """Base class for an invalid fingerprint or trajectory."""


class SchemaExtensionRequired(CategoricalFingerprintError):
    """A source needs a new schema value or component before it is complete."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _require_category(value: object, field: str) -> str:
    if not isinstance(value, str) or not _CATEGORY.fullmatch(value):
        raise CategoricalFingerprintError(
            f"{field} must be one lower-case categorical label with no digits: "
            f"{value!r}"
        )
    return value


def _schema_components(schema: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    components = schema.get("components")
    if not isinstance(components, list) or not components:
        raise CategoricalFingerprintError(
            "Campaign schema must declare a nonempty components list"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for component in components:
        if not isinstance(component, Mapping):
            raise CategoricalFingerprintError("A schema component must be an object")
        identifier = _require_category(component.get("id"), "schema component id")
        if identifier in result:
            raise CategoricalFingerprintError(
                f"Duplicate schema component: {identifier}"
            )
        values = component.get("values")
        if not isinstance(values, list) or not values:
            raise CategoricalFingerprintError(
                f"Component {identifier} must declare values"
            )
        normalized = [_require_category(item, f"{identifier} value") for item in values]
        if len(normalized) != len(set(normalized)):
            raise CategoricalFingerprintError(
                f"Component {identifier} has duplicate values"
            )
        if ABSENT not in normalized:
            raise CategoricalFingerprintError(
                f"Component {identifier} must permit {ABSENT}"
            )
        result[identifier] = component
    return result


def validate_campaign_schema(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a schema revision without treating its vocabulary as frozen.

    The registry itself is mutable.  This validator makes an undeclared value a
    visible extension request instead of collapsing it into an ``other`` bin.
    """
    _require_category(schema.get("id"), "campaign id")
    _schema_components(schema)
    return schema


def load_contract() -> Mapping[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("schema_version") != FINGERPRINT_VERSION:
        raise CategoricalFingerprintError(
            "Unexpected categorical fingerprint contract version"
        )
    if contract.get("absent_value") != ABSENT:
        raise CategoricalFingerprintError(
            "The contract must use the canonical absent value"
        )
    policy = contract.get("campaign_schema_policy", {})
    if policy.get("status") != "living":
        raise CategoricalFingerprintError(
            "Campaign schemas must remain living registries"
        )
    campaigns = contract.get("campaigns")
    if not isinstance(campaigns, list) or not campaigns:
        raise CategoricalFingerprintError("Contract has no campaign schemas")
    identifiers = []
    for campaign in campaigns:
        validate_campaign_schema(campaign)
        identifiers.append(campaign["id"])
    if len(identifiers) != len(set(identifiers)):
        raise CategoricalFingerprintError("Contract has duplicate campaign schemas")
    return contract


def campaign_schema(campaign_id: str) -> Mapping[str, Any]:
    _require_category(campaign_id, "campaign id")
    for schema in load_contract()["campaigns"]:
        if schema["id"] == campaign_id:
            return deepcopy(schema)
    raise KeyError(f"Unknown categorical-fingerprint campaign: {campaign_id}")


def schema_revision(schema: Mapping[str, Any]) -> str:
    """Content address for the exact living-registry revision used in an output."""
    validate_campaign_schema(schema)
    return "schema_" + sha256(_canonical_json(schema).encode("ascii")).hexdigest()[:16]


def validate_fingerprint(
    fingerprint: Mapping[str, object], schema: Mapping[str, Any]
) -> dict[str, str]:
    """Return a canonical, complete numeric-free categorical fingerprint."""
    components = _schema_components(schema)
    if not isinstance(fingerprint, Mapping):
        raise CategoricalFingerprintError("Fingerprint must be an object")
    expected, actual = set(components), set(fingerprint)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise CategoricalFingerprintError(
            "Fingerprint keys must exactly equal the campaign schema; "
            f"missing={missing}, extra={extra}"
        )
    normalized: dict[str, str] = {}
    for identifier in sorted(components):
        value = _require_category(fingerprint[identifier], identifier)
        permitted = set(components[identifier]["values"])
        if value not in permitted:
            raise SchemaExtensionRequired(
                f"{schema['id']}.{identifier}={value!r} is not in the living registry; "
                "extend the schema and reproject the whole campaign"
            )
        normalized[identifier] = value
    return normalized


def fingerprint_key(
    fingerprint: Mapping[str, object], schema: Mapping[str, Any]
) -> str:
    return _canonical_json(validate_fingerprint(fingerprint, schema))


def family_id(fingerprint: Mapping[str, object], schema: Mapping[str, Any]) -> str:
    """Stable display identifier for one exact categorical component vector."""
    encoded = fingerprint_key(fingerprint, schema).encode("ascii")
    return "family_" + sha256(encoded).hexdigest()[:16]


def changed_components(
    parent: Mapping[str, object], child: Mapping[str, object], schema: Mapping[str, Any]
) -> list[str]:
    before, after = (
        validate_fingerprint(parent, schema),
        validate_fingerprint(child, schema),
    )
    return [
        component
        for component in sorted(before)
        if before[component] != after[component]
    ]


def _require_record(
    record: Mapping[str, Any],
) -> tuple[str, int, str | None, dict[str, str]]:
    candidate_id = record.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise CategoricalFingerprintError("Each record needs a nonempty candidate_id")
    proposal = record.get("proposal")
    if type(proposal) is not int or proposal < 0:
        raise CategoricalFingerprintError(
            "Each record needs a nonnegative integer proposal"
        )
    parent_id = record.get("ontology_parent_id")
    if parent_id is not None and (not isinstance(parent_id, str) or not parent_id):
        raise CategoricalFingerprintError(
            "ontology_parent_id must be a nonempty string or null"
        )
    return candidate_id, proposal, parent_id, dict(record.get("fingerprint", {}))


def annotate_run(
    records: Iterable[Mapping[str, Any]], schema: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Attach the eight marginal/cumulative metrics to one chronological run.

    ``records`` must include the seed (proposal zero) and use the campaign's
    recorded first parent as ``ontology_parent_id``.  Component-state novelty
    is historical within the whole run, not just the direct ancestry path.
    """
    validate_campaign_schema(schema)
    prepared: list[tuple[Mapping[str, Any], str, int, str | None, dict[str, str]]] = []
    for record in records:
        candidate_id, proposal, parent_id, raw_fingerprint = _require_record(record)
        prepared.append(
            (
                record,
                candidate_id,
                proposal,
                parent_id,
                validate_fingerprint(raw_fingerprint, schema),
            )
        )
    prepared.sort(key=lambda item: (item[2], item[1]))
    if not prepared:
        return []

    by_id: dict[str, tuple[Mapping[str, Any], dict[str, str]]] = {}
    seen_values = {component: set() for component in _schema_components(schema)}
    seen_families: set[str] = set()
    totals = {name: 0 for name in METRICS if name.endswith("_cumulative")}
    annotated: list[dict[str, Any]] = []
    seen_proposals: set[int] = set()

    for original, candidate_id, proposal, parent_id, fingerprint in prepared:
        if proposal in seen_proposals:
            raise CategoricalFingerprintError(f"Duplicate proposal in run: {proposal}")
        seen_proposals.add(proposal)
        if not annotated and (proposal != 0 or parent_id is not None):
            raise CategoricalFingerprintError(
                "A run must start with its proposal-zero seed"
            )
        if candidate_id in by_id and by_id[candidate_id][1] != fingerprint:
            raise CategoricalFingerprintError(
                f"Repeated candidate_id has inconsistent fingerprints: {candidate_id}"
            )
        family_key = _canonical_json(fingerprint)
        if parent_id is None:
            if by_id:
                raise CategoricalFingerprintError(
                    "Only the seed may omit ontology_parent_id "
                    f"(candidate {candidate_id})"
                )
            changed: list[str] = []
            component_edits = component_novelty = family_switch = new_family = 0
        else:
            if parent_id not in by_id:
                raise CategoricalFingerprintError(
                    f"Primary parent {parent_id} must occur earlier in the same run"
                )
            parent_fingerprint = by_id[parent_id][1]
            changed = [
                key
                for key in sorted(fingerprint)
                if fingerprint[key] != parent_fingerprint[key]
            ]
            component_edits = len(changed)
            component_novelty = sum(
                fingerprint[key] not in seen_values[key] for key in changed
            )
            family_switch = int(bool(changed))
            new_family = int(bool(family_switch and family_key not in seen_families))

        totals["component_edits_cumulative"] += component_edits
        totals["new_component_states_cumulative"] += component_novelty
        totals["family_switches_cumulative"] += family_switch
        totals["new_families_cumulative"] += new_family
        metrics = {
            "component_edits_marginal": component_edits,
            "component_edits_cumulative": totals["component_edits_cumulative"],
            "new_component_states_marginal": component_novelty,
            "new_component_states_cumulative": totals[
                "new_component_states_cumulative"
            ],
            "family_switches_marginal": family_switch,
            "family_switches_cumulative": totals["family_switches_cumulative"],
            "new_families_marginal": new_family,
            "new_families_cumulative": totals["new_families_cumulative"],
        }
        row = deepcopy(dict(original))
        row.update(
            fingerprint=fingerprint,
            family_id=family_id(fingerprint, schema),
            family_components=deepcopy(fingerprint),
            changed_components=changed,
            metrics=metrics,
            schema_revision=schema_revision(schema),
            fingerprint_version=FINGERPRINT_VERSION,
        )
        annotated.append(row)
        by_id[candidate_id] = (row, fingerprint)
        seen_families.add(family_key)
        for component, value in fingerprint.items():
            seen_values[component].add(value)
    return annotated


def _materialize_canonical_seed_fingerprints(
    runs: Mapping[str, Iterable[Mapping[str, Any]]],
    schema: Mapping[str, Any],
    canonical_seed_fingerprints: Mapping[str, Mapping[str, object]] | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, str]]]:
    """Apply one reviewed seed fingerprint to every run that references it."""
    seed_fingerprints = canonical_seed_fingerprints or {}
    prepared_runs: dict[str, list[dict[str, Any]]] = {}
    used_seed_fingerprints: dict[str, dict[str, str]] = {}
    for run_id, records in sorted(runs.items()):
        prepared_records: list[dict[str, Any]] = []
        for record in records:
            row = deepcopy(dict(record))
            canonical_seed_id = row.get("canonical_seed_id")
            if canonical_seed_id is not None:
                if (
                    not row.get("is_seed")
                    or row.get("proposal") != 0
                    or row.get("ontology_parent_id") is not None
                    or row.get("candidate_id") != canonical_seed_id
                ):
                    raise CategoricalFingerprintError(
                        "canonical_seed_id must match a proposal-zero seed record"
                    )
                if not isinstance(canonical_seed_id, str) or not canonical_seed_id:
                    raise CategoricalFingerprintError(
                        "canonical_seed_id must be a nonempty string"
                    )
                if canonical_seed_id not in seed_fingerprints:
                    raise CategoricalFingerprintError(
                        "No canonical seed fingerprint was provided for "
                        f"{canonical_seed_id}"
                    )
                fingerprint = validate_fingerprint(
                    seed_fingerprints[canonical_seed_id], schema
                )
                if "fingerprint" in row and row["fingerprint"] != fingerprint:
                    raise CategoricalFingerprintError(
                        "Run seed fingerprint disagrees with its canonical seed"
                    )
                row["fingerprint"] = fingerprint
                used_seed_fingerprints[canonical_seed_id] = fingerprint
            prepared_records.append(row)
        prepared_runs[run_id] = prepared_records
    return prepared_runs, used_seed_fingerprints


def output_document(
    campaign_id: str,
    runs: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    canonical_seed_fingerprints: Mapping[str, Mapping[str, object]] | None = None,
    source_reviews: Mapping[str, Mapping[str, Any]] | None = None,
    source_bundles: Mapping[str, Mapping[str, str]] | None = None,
    schema_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-contained, revision-bound campaign output document.

    An inventory may mark a shared proposal-zero source with
    ``canonical_seed_id``. Supply that ID's reviewed categorical fingerprint
    once through ``canonical_seed_fingerprints``; it is materialized into each
    run solely for trajectory calculation. Every unique candidate requires a
    source-bound role review; the low-level metric functions are not a
    publication path. Seeds and other identical candidate IDs are reviewed once.
    A campaign fork may supply its own evolving schema without changing the
    shared registry. Its campaign identity and study scope must remain identical.
    """
    from experiments.ontology_categorical_review import (
        REVIEW_VERSION,
        validate_review_schema,
        validate_source_review,
    )

    schema = campaign_schema(campaign_id)
    if schema_override is not None:
        for field in (
            "id",
            "source_campaign_key",
            "data_directory",
            "included_conditions",
            "proposal_limit",
        ):
            if schema_override.get(field) != schema.get(field):
                raise CategoricalFingerprintError(
                    f"Campaign schema override cannot change study scope: {field}"
                )
        schema = deepcopy(dict(schema_override))
    validate_review_schema(schema)
    materialized_runs, used_seed_fingerprints = (
        _materialize_canonical_seed_fingerprints(
            runs, schema, canonical_seed_fingerprints
        )
    )
    if campaign_id == "addition":
        from experiments.live_trajectory_dashboard import dashboard_run_visible

        if any(not dashboard_run_visible(run_id) for run_id in materialized_runs):
            raise CategoricalFingerprintError(
                "Addition run is excluded by dashboard configuration and study scope"
            )
    unique_fingerprints: dict[str, dict[str, str]] = {}
    for records in materialized_runs.values():
        for record in records:
            candidate_id, proposal, _parent, raw_fingerprint = _require_record(record)
            limit = schema.get("proposal_limit")
            if limit is not None and proposal > limit:
                raise CategoricalFingerprintError("Proposal is outside campaign scope")
            condition = record.get("condition")
            if condition is not None and condition not in schema["included_conditions"]:
                raise CategoricalFingerprintError("Condition is outside campaign scope")
            fingerprint = validate_fingerprint(raw_fingerprint, schema)
            if (
                candidate_id in unique_fingerprints
                and unique_fingerprints[candidate_id] != fingerprint
            ):
                raise CategoricalFingerprintError(
                    "The same candidate cannot have different fingerprints across runs"
                )
            unique_fingerprints[candidate_id] = fingerprint
    if (
        source_reviews is None
        or source_bundles is None
        or set(source_reviews) != set(unique_fingerprints)
        or set(source_bundles) != set(unique_fingerprints)
    ):
        raise CategoricalFingerprintError(
            "Publication requires source_reviews and source_bundles for exactly "
            "the unique candidates, including each canonical seed once"
        )
    review_audits = {
        candidate_id: validate_source_review(
            source_reviews[candidate_id],
            source_bundles[candidate_id],
            fingerprint,
            schema,
        )
        for candidate_id, fingerprint in unique_fingerprints.items()
    }
    annotated_runs = {
        run_id: annotate_run(records, schema)
        for run_id, records in materialized_runs.items()
    }
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "campaign": campaign_id,
        "schema_revision": schema_revision(schema),
        "schema": schema,
        "metric_definitions": METRICS,
        "family_rule": load_contract()["family_rule"],
        "primary_parent_rule": load_contract()["primary_parent_rule"],
        "canonical_seed_fingerprints": used_seed_fingerprints,
        "review_version": REVIEW_VERSION,
        "source_reviews": deepcopy(dict(source_reviews)),
        "review_audits": review_audits,
        "runs": annotated_runs,
    }
