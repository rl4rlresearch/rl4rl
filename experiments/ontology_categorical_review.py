"""Source-bound, role-specific review gates for categorical fingerprints.

This validates review evidence, not arbitrary Python program equivalence. A
reviewer still establishes dataflow and semantic roles; spelling alone is
never enough to select a component. Candidate source is parsed, never run.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Mapping
from hashlib import sha256
from typing import Any

from experiments.ontology_categorical_fingerprint import (
    ABSENT,
    CategoricalFingerprintError,
    SchemaExtensionRequired,
    schema_revision,
    validate_campaign_schema,
    validate_fingerprint,
)

REVIEW_VERSION = "role_evidence_v1"
EXCLUDED_DISPOSITIONS = {
    "unreachable",
    "numerical_setting",
    "implementation_only",
    "covered_by_composite",
}


def _error(message: str) -> None:
    raise CategoricalFingerprintError(message)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _error(f"{field} needs a nonempty explanation")
    return value


def validate_review_schema(schema: Mapping[str, Any]) -> None:
    """Every component and category must declare its semantic boundary."""
    validate_campaign_schema(schema)
    for component in schema["components"]:
        name = component["id"]
        for key in ("scope", "excludes", "absence_rule", "numeric_invariants"):
            _text(component.get(key), f"{name}.{key}")
        definitions = component.get("value_definitions")
        if not isinstance(definitions, Mapping) or set(definitions) != set(
            component["values"]
        ):
            _error(f"{name}: every category needs an exact definition")
        folded_definitions: set[str] = set()
        for value, definition in definitions.items():
            normalized = " ".join(
                _text(definition, f"{name}.{value}").split()
            ).casefold()
            if normalized in folded_definitions:
                _error(f"{name}: equivalent definitions need one canonical category")
            folded_definitions.add(normalized)
        lookup = {value.casefold(): value for value in component["values"]}
        aliases = component.get("aliases")
        if not isinstance(aliases, Mapping):
            _error(f"{name}: an explicit role-scoped alias map is required")
        for alias, target in aliases.items():
            normalized = _text(alias, f"{name} alias").strip().casefold()
            if target not in definitions:
                _error(f"{name}: alias {alias!r} names an undeclared category")
            if normalized in lookup and lookup[normalized] != target:
                _error(f"{name}: conflicting alias {alias!r}")
            lookup[normalized] = target
        if "none" in definitions:
            _error(f"{name}: use absent instead of a second absence category")
        provisional = component.get("provisional_values", [])
        if not isinstance(provisional, list) or not set(provisional) <= set(
            definitions
        ):
            _error(f"{name}: provisional values must belong to the registry")


def validate_role_consistency(
    fingerprint: Mapping[str, str], schema: Mapping[str, Any]
) -> None:
    """Check declared dependencies; these supplement source review."""
    for component in schema["components"]:
        if fingerprint[component["id"]] in component.get("provisional_values", []):
            raise SchemaExtensionRequired(
                f"{component['id']}: broad draft category requires refinement "
                "before publication"
            )
    modulation = fingerprint.get("feature_modulation", ABSENT)
    details = ("modulation_source", "modulation_transfer", "gate_hidden_activation")
    if modulation == ABSENT and any(
        fingerprint.get(name, ABSENT) != ABSENT for name in details
    ):
        _error("Gate detail is present but feature_modulation is absent")
    if modulation != ABSENT and any(
        fingerprint.get(name, ABSENT) == ABSENT
        for name in ("modulation_source", "modulation_transfer")
    ):
        _error("Feature modulation requires its control source and transfer law")
    for mechanism, detail in (
        ("gated_mlp", "feedforward_gate"),
        ("fixed_feature_basis", "feedforward_basis"),
    ):
        if detail in fingerprint and (
            (fingerprint.get("feedforward_mechanism") == mechanism)
            != (fingerprint[detail] != ABSENT)
        ):
            _error(f"{detail} disagrees with feedforward_mechanism")
    if fingerprint.get("attention_topology") == ABSENT and any(
        fingerprint.get(name, ABSENT) != ABSENT
        for name in (
            "attention_mask",
            "attention_score_features",
            "attention_position_bias",
            "attention_weighting",
            "key_value_parameterization",
            "key_value_memory",
        )
    ):
        _error("Attention details are present but attention_topology is absent")


def canonicalize_observations(
    component: Mapping[str, Any], observed_values: Iterable[str]
) -> str:
    """Normalize equivalent observations only AFTER their role is established.

    A heterogeneous set cannot become a joined label or a 'primary' winner.
    Split its semantic roles in the living schema, then reproject the campaign.
    """
    if isinstance(observed_values, (str, bytes)):
        _error("Observations must be a list, not a combined label")
    lookup = {value.casefold(): value for value in component["values"]}
    lookup.update(
        {
            alias.strip().casefold(): value
            for alias, value in component["aliases"].items()
        }
    )
    resolved = set()
    for observation in observed_values:
        key = _text(observation, "observation").strip().casefold()
        if key not in lookup:
            raise SchemaExtensionRequired(
                f"{component['id']}: unrecognized observation {observation!r}; "
                "review its role and define an explicit equivalence or schema extension"
            )
        resolved.add(lookup[key])
    if len(resolved) != 1:
        raise SchemaExtensionRequired(
            f"{component['id']}: expected one mechanism for this role, got "
            f"{sorted(resolved)}; split roles or define the composition "
            "before publication"
        )
    return resolved.pop()


def source_digest(sources: Mapping[str, str]) -> str:
    return sha256(
        json.dumps(dict(sources), sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def source_inventory(sources: Mapping[str, str]) -> list[dict[str, Any]]:
    """List every Python statement for explicit coverage, including dead code.

    Ownership is reviewed at statement granularity. Multiple independently
    classified aspects can cite the same statement (e.g. operator and mask).
    Statement IDs are evidence only and never enter fingerprint values.
    """
    if not sources or any(
        not isinstance(path, str) or not isinstance(source, str)
        for path, source in sources.items()
    ):
        _error("Review requires the complete candidate source bundle")
    result = []
    for path, source in sorted(sources.items()):
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            _error(f"Source cannot be reviewed as executable Python: {exc}")
        for index, node in enumerate(ast.walk(tree)):
            if isinstance(node, ast.stmt):
                result.append(
                    {
                        "id": f"{path}:{node.lineno}:{node.col_offset}:{index}",
                        "file": path,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "kind": type(node).__name__,
                        "code": ast.get_source_segment(source, node),
                    }
                )
    if not result:
        _error("Source bundle has no Python statements to review")
    return result


def validate_source_review(
    review: Mapping[str, Any],
    sources: Mapping[str, str],
    fingerprint: Mapping[str, object],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject missing coverage, role collisions, stale evidence and lost facts."""
    validate_review_schema(schema)
    expected = validate_fingerprint(fingerprint, schema)
    validate_role_consistency(expected, schema)
    if review.get("version") != REVIEW_VERSION:
        _error("A role_evidence_v1 source review is required for publication")
    if review.get("schema_revision") != schema_revision(schema):
        _error("Review schema revision is stale; reproject and review the source")
    if review.get("source_sha256") != source_digest(sources):
        _error("Review source digest is stale or mismatched")
    _text(review.get("reviewer"), "reviewer")
    _text(review.get("entrypoint_analysis"), "entrypoint_analysis")
    statements = {item["id"]: item for item in source_inventory(sources)}
    components = {item["id"]: item for item in schema["components"]}
    component_reviews = review.get("components")
    if not isinstance(component_reviews, Mapping) or set(component_reviews) != set(
        components
    ):
        _error("Every component needs a role review, including absent components")
    coverage = review.get("statement_coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != set(statements):
        _error("Source statement coverage must be complete with no unknown IDs")
    facts = review.get("facts")
    if not isinstance(facts, list):
        _error("Review requires a semantic fact ledger")
    fact_map: dict[str, Mapping[str, Any]] = {}
    ownership: set[tuple[tuple[str, ...], str]] = set()
    for fact in facts:
        if not isinstance(fact, Mapping):
            _error("Each semantic fact must be an object")
        fact_id = _text(fact.get("id"), "fact.id")
        if fact_id in fact_map:
            _error(f"Duplicate semantic fact {fact_id}")
        owner = fact.get("component")
        if owner not in components:
            _error(f"Fact {fact_id} has no declared component owner")
        aspect = _text(fact.get("aspect"), f"{fact_id}.aspect")
        evidence = fact.get("statements")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(item not in statements for item in evidence)
        ):
            _error(f"Fact {fact_id} must cite current source statement IDs")
        key = (tuple(sorted(set(evidence))), aspect.strip().casefold())
        if key in ownership:
            _error(f"Duplicate ownership of the same semantic aspect: {fact_id}")
        ownership.add(key)
        _text(fact.get("role_reason"), f"{fact_id}.role_reason")
        values = fact.get("observations")
        if not isinstance(values, list):
            _error(f"Fact {fact_id} needs observed mechanisms")
        value = canonicalize_observations(components[owner], values)
        if value == ABSENT or value != expected[owner]:
            _error(f"Fact {fact_id} contradicts its component fingerprint")
        fact_map[fact_id] = fact
    used_facts: set[str] = set()
    for name, evidence in component_reviews.items():
        if not isinstance(evidence, Mapping):
            _error(f"{name}: role review must be an object")
        _text(evidence.get("reason"), f"{name}.reason")
        refs = evidence.get("facts")
        if not isinstance(refs, list) or len(set(refs)) != len(refs):
            _error(f"{name}: fact references must be a unique list")
        if expected[name] == ABSENT:
            if refs:
                _error(f"{name}: absent component cannot own an active fact")
        elif not refs:
            _error(f"{name}: non-absent component has no source evidence")
        for ref in refs:
            if ref not in fact_map or fact_map[ref]["component"] != name:
                _error(f"{name}: a semantic fact must have exactly one component owner")
            used_facts.add(ref)
    if used_facts != set(fact_map):
        _error(
            "Unassigned semantic facts: a mechanism was omitted from the fingerprint"
        )
    covered_facts: set[str] = set()
    for statement_id, disposition in coverage.items():
        if not isinstance(disposition, Mapping):
            _error(f"{statement_id}: coverage needs a disposition object")
        kind = disposition.get("kind")
        _text(disposition.get("reason"), f"{statement_id}.reason")
        refs = disposition.get("facts", [])
        if not isinstance(refs, list) or any(ref not in fact_map for ref in refs):
            _error(f"{statement_id}: coverage cites an unknown fact")
        if kind == "mechanism":
            if not refs or any(
                statement_id not in fact_map[ref]["statements"] for ref in refs
            ):
                _error(f"{statement_id}: mechanism coverage needs matching evidence")
        elif kind in EXCLUDED_DISPOSITIONS:
            if refs and kind != "covered_by_composite":
                _error(f"{statement_id}: excluded code cannot own active facts")
            if kind == "covered_by_composite" and not refs:
                _error(f"{statement_id}: composite exclusion needs an owning fact")
        else:
            _error(f"{statement_id}: unresolved or unknown coverage disposition")
        covered_facts.update(refs)
    if covered_facts != set(fact_map):
        _error("Semantic facts missing from source coverage")
    # Every cited statement must acknowledge its fact; a broad exclusion cannot
    # hide a contradictory active interpretation elsewhere in the certificate.
    for fact_id, fact in fact_map.items():
        for statement_id in fact["statements"]:
            if fact_id not in coverage[statement_id].get("facts", []):
                _error(f"{fact_id}: cited statement was excluded or has another owner")
    return {
        "version": REVIEW_VERSION,
        "source_sha256": review["source_sha256"],
        "schema_revision": review["schema_revision"],
        "statement_count": len(statements),
        "fact_count": len(facts),
        "component_count": len(components),
    }
