"""Guard the finite staged Fashion source-review batch against generic profiles."""
import json
from pathlib import Path

from experiments.review_ontology_sources import CORE, TASK_KEYS
from experiments.generate_fashion_candidate_catalog import ROOT, event, source
from experiments.ontology_review_vision_lm import FASHION, _staged_fashion_candidate_pair, profile, review_pair


CATALOG = Path(__file__).parents[1] / "experiments/ontology_fashion_candidate_references_v1.json"


def references():
    return {item["candidate_id"]: item for item in json.loads(CATALOG.read_text(encoding="utf8"))["reviewed_profiles"]}


def test_candidate_batch_is_complete_source_and_parent_bound():
    refs = references()
    assert len(refs) == 10
    required = CORE + TASK_KEYS["fashion"].split()
    for ref in refs.values():
        assert len(ref["fingerprint"]) == 27
        assert all(ref["fingerprint"][key] for key in required)
        assert ref["program_parts"] and ref["source_sha256"] and ref["program_sha256"]
        evidence = ref["evidence"][0]
        assert evidence["source_sha256"] == ref["source_sha256"]
        assert evidence["parent_source_sha256"] == ref["parent_source_sha256"]
        assert evidence["program_sha256"] == ref["program_sha256"]


def test_changed_profiles_name_their_source_bearing_mechanisms():
    refs = references()
    expected = {
        "a039e798": ("spatial_readout", "average/max"),
        "6cf043a9": ("spatial_downsampling", "PixelUnshuffle"),
        "d31132a4": ("channel_interaction", "covariance"),
        "75fa4635": ("routing", "query-key attention"),
        "2735b636": ("routing", "row-plus-column"),
        "1fd68600": ("routing", "Conv1d channel recalibration"),
        "6260a636": ("spatial_readout", "learned vertical"),
        "d2267780": ("routing", "tanh"),
    }
    for prefix, (field, mechanism) in expected.items():
        ref = next(item for candidate, item in refs.items() if candidate.startswith(prefix))
        assert ref["transition_classification"] == "changing"
        assert mechanism in ref["fingerprint"][field]


def test_residual_cnn_replacement_is_a_source_bound_ontology_change():
    refs = references()
    replacement = refs[next(key for key in refs if key.startswith("45ca5bf9"))]
    assert replacement["transition_classification"] == "changing"
    assert set(replacement["changed_components"]) == {
        "normalization", "activation", "connectivity", "aggregation",
        "scale_representation", "spatial_readout",
    }
    assert "fixed additive residual" in replacement["fingerprint"]["connectivity"]


def test_preserving_entries_still_name_their_actual_lifecycle_work():
    refs = references()
    assert "running-stat buffers are explicitly disabled" in refs[next(key for key in refs if key.startswith("5392d8bb"))]["fingerprint"]["state"]


def test_staged_decision_requires_the_exact_recorded_parent_and_preserves_component_witness():
    refs = references()
    # One representative source proves the whole candidate catalog stays out of
    # standalone admission; the following loop independently checks every
    # source-pair declaration and its exact parent binding.
    sample_id, sample = next(iter(refs.items()))
    assert profile(source(ROOT / sample["reference_run"] / "candidates" / sample_id), FASHION) is None
    for candidate_id, ref in refs.items():
        before = source(ROOT / ref["reference_run"] / "candidates" / ref["parent_candidate_id"])
        after = source(ROOT / ref["reference_run"] / "candidates" / candidate_id)
        result = review_pair(before, after, FASHION)
        assert result["classification"] == ref["transition_classification"]
        assert result["parent_source_sha256"] == ref["parent_source_sha256"]
        assert result["changed_components"] == ref["changed_components"]
        assert result["evidence"][-1]["kind"] == "exact staged candidate-parent transition declaration"
        assert _staged_fashion_candidate_pair(after, after) is None
