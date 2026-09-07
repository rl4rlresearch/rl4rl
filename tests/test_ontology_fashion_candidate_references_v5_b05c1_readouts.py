"""Regression coverage for the finite B05-C1 supplementary-readout catalog."""
from experiments.generate_fashion_candidate_catalog_v2 import ROOT, source
from experiments.ontology_fashion_candidate_references_v5_b05c1_readouts import SPECS, profiles
from experiments.ontology_review_vision_lm import _staged_fashion_candidate_pair, digest
from experiments.review_ontology_sources import CORE, TASK_KEYS


def test_b05_c1_supplementary_readouts_are_complete_exact_parent_pairs():
    refs = profiles()
    assert len(refs) == len(SPECS) == 3
    assert {ref["reference_proposal"] for ref in refs} == {51, 64, 100}
    required = CORE + TASK_KEYS["fashion"].split()
    for ref in refs:
        child = source(ROOT / ref["reference_run"] / "candidates" / ref["candidate_id"])
        parent = source(ROOT / ref["reference_run"] / "candidates" / ref["parent_candidate_id"])
        assert digest(child) == ref["source_sha256"]
        assert digest(parent) == ref["parent_source_sha256"]
        assert ref["transition_classification"] == "changing"
        assert all(ref["fingerprint"][field] for field in required)
        evidence = ref["evidence"][0]
        assert evidence["directed_parent_source_sha256"] == ref["parent_source_sha256"]
        assert evidence["retained_parent_candidate_id"] == ref["parent_candidate_id"]


def test_b05_c1_admission_rejects_every_nonidentical_child_or_parent():
    for ref in profiles():
        child = source(ROOT / ref["reference_run"] / "candidates" / ref["candidate_id"])
        parent = source(ROOT / ref["reference_run"] / "candidates" / ref["parent_candidate_id"])
        admitted = _staged_fashion_candidate_pair(parent, child)
        assert admitted and admitted["classification"] == "changing"
        assert admitted["changed_components"] == ref["changed_components"]
        assert admitted["fingerprint"] == ref["fingerprint"]
        assert _staged_fashion_candidate_pair(child, child) is None
        assert _staged_fashion_candidate_pair(parent, {"train.py": child["train.py"] + "\n# mutation"}) is None

def test_b05_c1_catalog_has_no_mutable_live_queue_dependency():
    import experiments.ontology_fashion_candidate_references_v5_b05c1_readouts as catalog
    before = catalog.profiles()
    text = __import__('pathlib').Path(catalog.__file__).read_text()
    assert 'outputs/ontology/live/queue' not in text
    assert 'WORKSPACE' not in text
    assert catalog.profiles() == before
