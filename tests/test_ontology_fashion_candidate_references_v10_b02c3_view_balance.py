from experiments.generate_fashion_candidate_catalog_v2 import ROOT,source
from experiments.ontology_fashion_candidate_references_v10_b02c3_view_balance import profiles,SHA
from experiments.ontology_review_vision_lm import _staged_fashion_candidate_pair,digest
from experiments.review_ontology_sources import CORE,TASK_KEYS
def test_p84_is_complete_static_exact_pair():
 r=profiles()[0];c=source(ROOT/r['reference_run']/'candidates'/r['candidate_id']);p=source(ROOT/r['reference_run']/'candidates'/r['parent_candidate_id']);assert digest(c)==SHA and digest(p)==r['parent_source_sha256'];assert all(r['fingerprint'][k] for k in CORE+TASK_KEYS['fashion'].split());assert _staged_fashion_candidate_pair(p,c)['classification']=='preserving'
def test_p84_rejects_mutation_and_has_no_live_queue_dependency():
 r=profiles()[0];c=source(ROOT/r['reference_run']/'candidates'/r['candidate_id']);p=source(ROOT/r['reference_run']/'candidates'/r['parent_candidate_id']);assert _staged_fashion_candidate_pair(c,c) is None;assert _staged_fashion_candidate_pair(p,{'train.py':c['train.py']+'\n# mutation'}) is None;import pathlib,experiments.ontology_fashion_candidate_references_v10_b02c3_view_balance as m;assert 'outputs/ontology/live/queue' not in pathlib.Path(m.__file__).read_text();assert profiles()==profiles()



