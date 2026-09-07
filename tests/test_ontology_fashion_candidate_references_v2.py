import copy
import json
from pathlib import Path

from experiments.generate_fashion_candidate_catalog_v2 import WORKSPACE, source
from experiments.ontology_review_vision_lm import FASHION, digest, review_pair
from experiments.ontology_review_service import reconcile_families
from experiments.review_ontology_sources import CORE, TASK_KEYS


CATALOG = Path(__file__).parents[1] / "experiments/ontology_fashion_candidate_references_v2.json"
CAMPAIGN_RUNS = Path("data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign/runs")


def campaign_runs() -> Path:
    """Find source data from the copied engine, independent of its cwd."""
    for directory in Path(__file__).resolve().parents:
        runs = directory / CAMPAIGN_RUNS
        if runs.is_dir():
            return runs
    raise FileNotFoundError(f"could not locate {CAMPAIGN_RUNS} above {__file__}")


ROOT = campaign_runs()


def references():
    return json.loads(CATALOG.read_text(encoding="utf8"))["reviewed_profiles"]


def test_second_batch_has_complete_exact_parent_pairs():
    refs = references()
    required = CORE + TASK_KEYS["fashion"].split()
    assert len(refs) == 10
    for ref in refs:
        assert len(ref["fingerprint"]) == 27
        assert all(ref["fingerprint"][key] for key in required)
        candidate = source(ROOT / ref["reference_run"] / "candidates" / ref["candidate_id"])
        parent = source(ROOT / ref["reference_run"] / "candidates" / ref["parent_candidate_id"])
        assert digest(candidate) == ref["source_sha256"]
        assert digest(parent) == ref["parent_source_sha256"]
        evidence = ref["evidence"][0]
        assert evidence["source_sha256"] == ref["source_sha256"]
        assert evidence["parent_source_sha256"] == ref["parent_source_sha256"]
        assert evidence["program_sha256"] == ref["program_sha256"]
        assert evidence["changed_components"] == ref["changed_components"]
        result = review_pair(parent, candidate, FASHION)
        assert result["classification"] == ref["transition_classification"]
        assert result["changed_components"] == ref["changed_components"]
        assert result["source_sha256"] == ref["source_sha256"]
        assert result["parent_source_sha256"] == ref["parent_source_sha256"]
        assert review_pair(candidate, candidate, FASHION) is None


def test_lifecycle_only_pairs_retain_their_distinct_gate_family_signatures():
    refs = references()
    mean_gate = next(ref for ref in refs if ref["candidate_id"].startswith("b951baa7"))
    topk_gate = next(ref for ref in refs if ref["candidate_id"].startswith("ddb6ff1b"))
    assert mean_gate["transition_classification"] == topk_gate["transition_classification"] == "preserving"
    assert mean_gate["family_signature"]["channel_gate"]["basis"] == ["mean", "maximum"]
    assert topk_gate["family_signature"]["channel_gate"]["basis"] == ["mean", "top-k mean"]
    assert "top-k spatial mean" in topk_gate["fingerprint"]["routing"]


def test_gate_signature_correction_removes_each_existing_b02_c2_conflict_without_erasing_changes():
    document = json.loads((WORKSPACE / "outputs/ontology/openevolve_v21_fashion_mnist.json").read_text())
    rows = copy.deepcopy(document["rows"])
    conflicts = [row for row in rows if row["classification"] == "uncertain"
                 and row["notes"].startswith("Family consistency")]
    # The mutable audit may already contain the corrected signatures.  When
    # it retains the pre-correction rows, every one originated as changing.
    assert {row["classification_before_consistency_check"] for row in conflicts} <= {"changing"}
    corrected = {ref["source_sha256"]: ref for ref in references()
                 if ref["candidate_id"].startswith(("b951baa7", "ddb6ff1b"))}
    for row in rows:
        ref = corrected.get(row.get("source_sha256"))
        if ref:
            row["family_signature"] = ref["family_signature"]
            row["fingerprint"] = ref["fingerprint"]
            row["fingerprint_complete"] = True
    historical = set(corrected)
    assert len(reconcile_families([row for row in rows if row.get("source_sha256") in historical])) == 0


