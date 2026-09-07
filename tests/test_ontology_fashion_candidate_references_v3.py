"""Regression coverage for queue-bound B03-C3, B05-C1, and parallel-branch Fashion profiles."""
import json
from pathlib import Path

from experiments.generate_fashion_candidate_catalog_v2 import source
from experiments.generate_fashion_candidate_catalog_v3 import PARENTS, ROOT, RUN
from experiments.ontology_review_vision_lm import FASHION, _staged_fashion_candidate_pair, digest, review_pair
from experiments.review_ontology_sources import CORE, TASK_KEYS


CATALOG = Path(__file__).parents[1] / "experiments/ontology_fashion_candidate_references_v3_unresolved.json"


def references():
    return json.loads(CATALOG.read_text(encoding="utf-8"))["reviewed_profiles"]


def test_readable_residuals_are_bound_to_queue_parent_sources():
    refs = references()
    # The original 14 queue-bound residuals, seven B03-C3 fixed-TTA reviews,
    # two B05-C1 exact adaptive-pooling parent/child reviews, and two B03-C3 parallel-branch parent/child reviews.
    assert len(refs) == len(PARENTS) + 7 + 2 + 2 == 25
    required = CORE + TASK_KEYS["fashion"].split()
    for ref in refs:
        child = source(ROOT / ref["reference_run"] / "candidates" / ref["candidate_id"])
        parent = source(ROOT / ref["reference_run"] / "candidates" / ref["parent_candidate_id"])
        assert digest(child) == ref["source_sha256"]
        assert digest(parent) == ref["parent_source_sha256"]
        assert len(ref["fingerprint"]) == 27
        assert all(ref["fingerprint"][field] for field in required)
        evidence = next(item for item in ref["evidence"]
                        if "retained_parent_candidate_id" in item
                        or "before_source_sha256" in item)
        if "retained_parent_candidate_id" in evidence:
            assert evidence["retained_parent_candidate_id"] == ref["parent_candidate_id"]
            assert evidence["queue_parent_source_sha256"] == [ref["parent_source_sha256"]]
        else:
            assert evidence["before_source_sha256"] == ref["parent_source_sha256"]
            assert evidence["after_source_sha256"] == ref["source_sha256"]


def test_exact_queue_bound_pairs_replay_and_no_nearby_pair_is_admitted():
    for ref in references():
        child = source(ROOT / ref["reference_run"] / "candidates" / ref["candidate_id"])
        parent = source(ROOT / ref["reference_run"] / "candidates" / ref["parent_candidate_id"])
        result = review_pair(parent, child, FASHION)
        assert result and result["classification"] == ref["transition_classification"]
        assert result["changed_components"] == ref["changed_components"]
        assert result["fingerprint"] == ref["fingerprint"]
        # Catalog admission is limited to the recorded directed parent-child pair.
        # A source may be independently parsable, so test the staged catalog gate itself.
        assert _staged_fashion_candidate_pair(child, child) is None


def test_only_traced_coordinate_gating_and_new_image_basis_cross_the_boundary():
    by_proposal = {ref["reference_proposal"]: ref for ref in references()
                   if ref["reference_run"] == RUN}
    assert {proposal for proposal, ref in by_proposal.items()
            if ref["transition_classification"] == "changing"} == {40, 121}
    assert by_proposal[40]["changed_components"] == ["routing", "channel_interaction"]
    assert "row and column" in by_proposal[40]["fingerprint"]["routing"]
    assert by_proposal[121]["changed_components"] == ["input_transform"]
    assert "broad local contrast" in by_proposal[121]["fingerprint"]["input_transform"]