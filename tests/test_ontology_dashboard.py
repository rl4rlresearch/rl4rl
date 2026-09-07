import json

import pytest

from experiments import live_trajectory_dashboard as dashboard


def test_source_diff_uses_recorded_parent_and_failed_patch_artifact(tmp_path):
    run = tmp_path / 'runs' / 'r'
    parent = run / 'candidates' / 'parent'
    child = run / 'candidates' / 'child'
    parent.mkdir(parents=True)
    child.mkdir()
    (parent / 'train.py').write_text('width = 8\n')
    (child / 'train.py').write_text('width = 4\n')
    events = [dict(event='proposal_completed', opportunity=1, candidate_id='child',
                   parent_ids=['parent'], artifact_path='candidates\\child'),
              dict(event='proposal_completed', opportunity=2, candidate_id='failed',
                   parent_ids=['parent'], artifact_path='candidates/parent')]
    (run / 'events.jsonl').write_text('\n'.join(json.dumps(row) for row in events))
    result = dashboard.ontology_source_payload(tmp_path, 'r', 1)
    assert '+width = 4' in result['parents'][0]['diff']
    assert dashboard.ontology_source_payload(tmp_path, 'r', 2)['parents'][0]['diff'] == ''
    (child / 'train.py').unlink()
    assert dashboard.ontology_source_payload(tmp_path, 'r', 1)['source_available'] is False
    with pytest.raises(ValueError):
        dashboard.ontology_source_payload(tmp_path, '../r', 1)
    with pytest.raises(ValueError):
        dashboard.ontology_source_payload(tmp_path, 'r', 999)


def test_source_rejects_recorded_artifact_escape(tmp_path):
    run = tmp_path / 'runs' / 'r'
    run.mkdir(parents=True)
    (tmp_path / 'secret.py').write_text('secret = 1')
    (run / 'events.jsonl').write_text(json.dumps(dict(
        event='proposal_completed', opportunity=1, artifact_path='../..', parent_ids=[])))
    result = dashboard.ontology_source_payload(tmp_path, 'r', 1)
    assert result['source'] == {}


def test_ontology_assets_are_watched_and_navigation_is_present():
    assert '/ontology' in dashboard.read_dashboard_page()
    assert '/ontology' in dashboard.read_science_page()
    assert '/ontology' in dashboard.read_transcript_page()
    assert '/ontology' in dashboard.read_controller_page()
    assert dashboard.ONTOLOGY_PAGE_PATH.is_file()


def test_saved_reviews_refresh_after_atomic_replacement(tmp_path):
    file = tmp_path / 'review.json'
    assert dashboard.ontology_review_response(file)['review'] is None
    file.write_text(json.dumps({'rows': [1]}))
    first = dashboard.ontology_review_response(file)
    assert first['review']['rows'] == [1]
    assert dashboard.ontology_review_response(file, first['revision'])['unchanged']
    replacement = tmp_path / 'replacement.json'
    replacement.write_text(json.dumps({'rows': [1, 2]}))
    replacement.replace(file)
    fresh = dashboard.ontology_review_response(file, first['revision'])
    assert fresh['review']['rows'] == [1, 2]
    assert fresh['revision'] != first['revision']
