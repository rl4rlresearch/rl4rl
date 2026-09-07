"""Regression coverage for the finite B04-C1 exact-parent catalog."""
from experiments.generate_fashion_candidate_catalog_v2 import ROOT, source
from experiments.ontology_fashion_candidate_references_v4_b04c1 import SPECS, profiles
from experiments.ontology_review_vision_lm import FASHION, _staged_fashion_candidate_pair, digest, review_pair
from experiments.review_ontology_sources import CORE, TASK_KEYS


def test_b04_c1_profiles_are_complete_and_static_parent_bound():
    refs = profiles()
    assert len(refs) == len(SPECS) == 10
    required = CORE + TASK_KEYS["fashion"].split()
    assert {ref["reference_proposal"]: ref["transition_classification"] for ref in refs} == {
        50: "preserving", 51: "changing", 91: "changing", 120: "changing", 124: "preserving",
        126: "preserving", 130: "changing", 136: "preserving", 192: "preserving", 196: "preserving",
    }
    for ref in refs:
        child = source(ROOT / ref["reference_run"] / "candidates" / ref["candidate_id"])
        parent = source(ROOT / ref["reference_run"] / "candidates" / ref["parent_candidate_id"])
        assert digest(child) == ref["source_sha256"]
        assert digest(parent) == ref["parent_source_sha256"]
        assert all(ref["fingerprint"][field] for field in required)
        evidence = ref["evidence"][0]
        assert evidence["directed_parent_source_sha256"] == ref["parent_source_sha256"]
        assert evidence["retained_parent_candidate_id"] == ref["parent_candidate_id"]


def test_b04_c1_admission_requires_the_exact_sha_bound_directed_pair():
    for ref in profiles():
        child = source(ROOT / ref["reference_run"] / "candidates" / ref["candidate_id"])
        parent = source(ROOT / ref["reference_run"] / "candidates" / ref["parent_candidate_id"])
        result = _staged_fashion_candidate_pair(parent, child)
        assert result and result["classification"] == ref["transition_classification"]
        assert result["changed_components"] == ref["changed_components"]
        assert result["fingerprint"] == ref["fingerprint"]
        assert _staged_fashion_candidate_pair(child, child) is None
        assert _staged_fashion_candidate_pair(parent, {"train.py": child["train.py"] + "\n# mutation"}) is None
        direct = review_pair(parent, child, FASHION)
        assert direct is None or direct["source_sha256"] == ref["source_sha256"]


def test_b04_c1_engine_catalog_has_no_mutable_live_queue_dependency():
    import experiments.ontology_fashion_candidate_references_v4_b04c1 as catalog
    before = catalog.profiles()
    text = __import__('pathlib').Path(catalog.__file__).read_text()
    assert 'outputs/ontology/live/queue' not in text
    assert 'WORKSPACE' not in text
    assert catalog.profiles() == before
