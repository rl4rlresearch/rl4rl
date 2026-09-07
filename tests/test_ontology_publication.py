import copy

import pytest

from experiments.ontology_seed_rubric import seed_fingerprint
from experiments.publish_ontology_reviews import validate


def fixture():
    campaign = {'campaign': 'test', 'runs': [{'run_id': 'r', 'points': [
        {'proposal': 1, 'candidate_id': 'a'}]}]}
    row = {'run_id': 'r', 'proposal': 1, 'candidate_id': 'a', 'classification': 'uncertain',
           'source_sha256': 'source', 'adjudication_code_sha256': 'engine',
           'fingerprint': {'embedding': 'lookup table'}, 'training': {}, 'inference': {}}
    return {'schema_version': '1.0', 'campaign': 'test', 'rows': [row]}, campaign


def test_every_record_processed_does_not_mean_complete():
    doc, campaign = fixture()
    summary, queue = validate(doc, campaign, 'addition')
    assert summary['rows'] == 1 and summary['resolved_transitions'] == 0
    assert not summary['fully_classified'] and len(queue) == 1
    doc['rows'][0]['fingerprint_complete'] = True
    with pytest.raises(ValueError, match='Partial fingerprint'):
        validate(doc, campaign, 'addition')


def test_complete_requires_all_components_and_provenance():
    doc, campaign = fixture()
    doc['rows'][0].update(classification='preserving', fingerprint_complete=True,
                         fingerprint=seed_fingerprint('openevolve_v21'))
    with pytest.raises(ValueError, match='provenance'):
        validate(doc, campaign, 'addition')
    doc['rows'][0]['fingerprint_provenance'] = {'kind': 'hash-locked audited seed'}
    assert validate(doc, campaign, 'addition')[0]['fully_classified']
    duplicate = copy.deepcopy(doc)
    duplicate['rows'] *= 2
    with pytest.raises(ValueError, match='Duplicate'):
        validate(duplicate, campaign, 'addition')


def test_invalid_source_requires_campaign_confirmation():
    doc, campaign = fixture()
    doc['rows'][0].update(classification='invalid_source', implemented=False, executable=False)
    with pytest.raises(ValueError, match='Invalid-source label lacks source'):
        validate(doc, campaign, 'addition')
    campaign['runs'][0]['points'][0].update(failure_kind='source_preflight', valid=False)
    assert validate(doc, campaign, 'addition')[0]['fully_classified']
