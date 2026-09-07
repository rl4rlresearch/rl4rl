from experiments.generate_fashion_candidate_catalog_v2 import ROOT, source
from experiments.ontology_fashion_candidate_references_v6_b02c3_crop_fusion import SPECS, profiles
from experiments.ontology_review_vision_lm import _staged_fashion_candidate_pair, digest
from experiments.review_ontology_sources import CORE, TASK_KEYS
def test_b02c3_crop_fusion_is_exact_and_complete():
 refs=profiles(); assert len(refs)==len(SPECS)==3; assert {r['reference_proposal'] for r in refs}=={139,142,148}
 for r in refs:
  c=source(ROOT/r['reference_run']/'candidates'/r['candidate_id']); p=source(ROOT/r['reference_run']/'candidates'/r['parent_candidate_id'])
  assert digest(c)==r['source_sha256'] and digest(p)==r['parent_source_sha256']; assert all(r['fingerprint'][k] for k in CORE+TASK_KEYS['fashion'].split())
def test_b02c3_crop_fusion_rejects_mutations():
 for r in profiles():
  c=source(ROOT/r['reference_run']/'candidates'/r['candidate_id']); p=source(ROOT/r['reference_run']/'candidates'/r['parent_candidate_id'])
  assert _staged_fashion_candidate_pair(p,c)['classification']=='changing'; assert _staged_fashion_candidate_pair(c,c) is None; assert _staged_fashion_candidate_pair(p,{'train.py':c['train.py']+'\n# mutation'}) is None

def test_b02c3_catalog_does_not_read_mutable_live_queue():
    import experiments.ontology_fashion_candidate_references_v6_b02c3_crop_fusion as catalog
    before = catalog.profiles()
    text = __import__('pathlib').Path(catalog.__file__).read_text()
    assert 'outputs/ontology/live/queue' not in text
    assert 'WORKSPACE' not in text
    assert catalog.profiles() == before

def test_b02c3_crop_fusion_family_signatures_reconcile_without_conflict():
    from experiments.ontology_review_service import reconcile_families
    rows = []
    for ref in profiles():
        rows.append({'source_sha256': ref['source_sha256'], 'classification': ref['transition_classification'], 'parent_source_sha256': [ref['parent_source_sha256']], 'fingerprint_complete': True, 'fingerprint': ref['fingerprint'], 'family_signature': ref['family_signature'], 'parent_reviews': []})
    assert reconcile_families(rows) == []
