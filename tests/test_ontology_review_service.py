import json
from pathlib import Path

import pytest

from experiments import ontology_review_service as service
from experiments.ontology_seed_rubric import seed_fingerprint


def test_family_reconciliation_unifies_proven_encodings_without_filling_fingerprints():
    seed = dict(source_sha256='a', classification='preserving', family='Automatic recurrent DAG',
                family_signature={'recurrence_dag': 'standard GRU'}, fingerprint_complete=True,
                fingerprint={'state': 'automatic description'})
    manual = dict(source_sha256='b', parent_source_sha256=['a'], classification='preserving',
                  family='Reviewed GRU', family_signature={'recurrence': 'standard GRU'},
                  fingerprint_complete=True, fingerprint={'state': 'reviewed description'})
    partial = dict(source_sha256='c', parent_source_sha256=['b'], classification='preserving',
                   fingerprint_complete=False, fingerprint={'state': None})
    same_signature = dict(source_sha256='d', classification='uncertain', family='Another label',
                          family_signature={'recurrence': 'standard GRU'}, fingerprint_complete=True)
    assert service.reconcile_families([seed, manual, partial, same_signature]) == []
    assert len({r['family'] for r in [seed, manual, partial, same_signature]}) == 1
    assert partial['fingerprint_complete'] is False and partial['fingerprint']['state'] is None
    assert seed['fingerprint'] != manual['fingerprint']


def test_family_reconciliation_respects_each_parent_and_queues_contradictions():
    def row(source, signature):
        return dict(source_sha256=source, classification='preserving',
                    family_signature=signature, family='mechanism', fingerprint_complete=True)
    before = row('a', {'memory': 'independent'})
    other = row('b', {'memory': 'shared'})
    after = row('c', {'manual_memory': 'shared'})
    after.update(classification='changing', parent_reviews=[
        {'source_sha256': 'a', 'classification': 'changing'},
        {'source_sha256': 'b', 'classification': 'preserving'}])
    assert service.reconcile_families([before, other, after]) == []
    assert before['family'] != after['family'] == other['family']
    contradiction = row('d', {'memory': 'independent'})
    contradiction.update(classification='changing', parent_source_sha256=['a'])
    conflicts = service.reconcile_families([before, contradiction])
    assert len(conflicts) == 1 and contradiction['classification'] == 'uncertain'
    assert contradiction['fingerprint_complete']


def test_family_reconciliation_does_not_invent_family_from_source_identity():
    pending = dict(source_sha256='a', classification='preserving', fingerprint_complete=False)
    assert service.reconcile_families([pending]) == []
    assert 'family' not in pending


def test_exact_source_review_replaces_a_stale_cached_fingerprint_without_merging_transitions():
    reviewed = dict(
        source_sha256='same-candidate-source', candidate_id='source-reviewed',
        classification='changing', fingerprint_complete=True,
        fingerprint={'mixing': 'source-traced program', 'state': 'no recurrence'},
        parent_reviews=[{'source_sha256': 'parent-a', 'classification': 'changing'}],
    )
    cached = dict(
        source_sha256='same-candidate-source', candidate_id='cached-copy',
        classification='preserving', fingerprint_complete=True,
        fingerprint={'mixing': 'obsolete generic description', 'state': 'no recurrence'},
        parent_reviews=[],
    )
    service.canonicalize_exact_source_fingerprints([reviewed, cached])
    assert reviewed['classification'] == 'changing'
    assert cached['classification'] == 'preserving'
    assert cached['fingerprint'] == reviewed['fingerprint']
    assert cached['fingerprint_provenance'] == {
        'kind': 'exact reviewed source match',
        'source_sha256': 'same-candidate-source',
        'anchor_candidate_id': 'source-reviewed',
    }


class InlinePool:
    def __init__(self, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def map(self, fn, jobs, **kwargs):
        return [dict(review=None) for job in jobs]


def test_source_cache_detects_added_helpers_and_modified_content(tmp_path):
    root = tmp_path / 'candidate'
    root.mkdir()
    file = root / 'train.py'
    file.write_text('x=1\n')
    cache = service.SourceCache(tmp_path / 'cache.sqlite')
    try:
        assert cache.read(root) == {'train.py': 'x=1\n'}
        file.write_text('x=234\n')
        (root / 'helper.py').write_text('y=4\n')
        assert cache.read(root) == {'train.py': 'x=234\n', 'helper.py': 'y=4\n'}
    finally:
        cache.close()


def test_process_lock_rejects_overlap_and_releases_after_close(tmp_path):
    lock = service.acquire_lock(tmp_path)
    try:
        with pytest.raises(service.ReviewBusy):
            service.acquire_lock(tmp_path)
    finally:
        lock.close()
    service.acquire_lock(tmp_path).close()


def test_added_proposals_are_classified_or_enter_the_durable_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'ProcessPoolExecutor', InlinePool)
    campaign = tmp_path / 'campaign'
    run = campaign / 'runs' / 'r'
    for candidate, code in [('seed', 'x=1\n'), ('same', 'x=1\n'), ('novel', 'x=2\n')]:
        directory = run / 'candidates' / candidate
        directory.mkdir(parents=True)
        (directory / 'train.py').write_text(code)
    key = 'openevolve_v21'
    seed = dict(proposal=0, candidate_id='seed', is_seed=True, valid=True, parent_ids=[])
    data = dict(available=True, campaign=str(campaign), runs=[dict(run_id='r', points=[seed])])
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'reviews'
    output.mkdir()
    row = service.blank_row(data['runs'][0], seed, {'train.py': 'x=1\n'}, 'test')
    row.update(classification='preserving', family='seed', fingerprint=seed_fingerprint(key),
               fingerprint_complete=True, fingerprint_provenance={'kind': 'test source reference'})
    (output / (key + '.json')).write_text(json.dumps(dict(rows=[row])))
    def run_once():
        snapshot.write_text(json.dumps(dict(campaigns={key: data})))
        return service.process_snapshot(snapshot, output, tmp_path / 'state', 'test', 1)
    assert run_once()['fully_classified']
    data['runs'][0]['points'].append(dict(proposal=1, candidate_id='same', parent_ids=['seed']))
    audit = run_once()
    assert audit['fully_classified'] and audit['campaigns'][key]['rows'] == 2
    data['runs'][0]['points'].append(dict(proposal=2, candidate_id='novel', parent_ids=['seed']))
    audit = run_once()
    assert not audit['transitions_complete'] and audit['campaigns'][key]['rows'] == 3
    queue = json.loads((tmp_path / 'state' / 'queue' / (key + '.json')).read_text())
    assert [row['candidate_id'] for row in queue] == ['novel']


def test_tiny_adderboard_is_outside_the_continuous_review_scope(tmp_path):
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps({'campaigns': {
        'tiny_adderboard_v21': {'available': True, 'campaign': str(tmp_path / 'tiny'), 'runs': []},
    }}))
    audit = service.process_snapshot(snapshot, tmp_path / 'reviews', tmp_path / 'state', 'test', 1)
    assert audit['campaigns'] == {}
    assert audit['pending_campaigns'] == []
    assert audit['excluded_campaigns'] == ['tiny_adderboard_v21']
    assert not (tmp_path / 'reviews' / 'tiny_adderboard_v21.json').exists()


def test_mismatched_artifact_identity_is_source_unavailable_not_a_parent_review(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'ProcessPoolExecutor', InlinePool)
    key = 'openevolve_v21'
    campaign = tmp_path / 'campaign'
    run = campaign / 'runs' / 'r'
    (run / 'candidates' / 'parent').mkdir(parents=True)
    (run / 'candidates' / 'parent' / 'train.py').write_text('x=1\n')
    point = dict(proposal=1, candidate_id='missing', artifact_path='candidates/parent',
                 parent_ids=['parent'], valid=True)
    data = dict(available=True, campaign=str(campaign), runs=[dict(run_id='r', points=[point])])
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps(dict(campaigns={key: data})))
    audit = service.process_snapshot(snapshot, tmp_path / 'reviews', tmp_path / 'state', 'test', 1)
    row = json.loads((tmp_path / 'reviews' / (key + '.json')).read_text())['rows'][0]
    assert row['classification'] == 'source_unavailable'
    assert row['source_unavailable_provenance']['candidate_id'] == 'missing'
    assert 'identity mismatch' in row['source_unavailable_provenance']['error']
    assert json.loads((tmp_path / 'state' / 'queue' / (key + '.json')).read_text()) == []
    assert audit['campaigns'][key]['fully_classified']
    assert audit['fully_classified']


def test_missing_recorded_parent_is_terminal_without_substituting_a_lineage_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'ProcessPoolExecutor', InlinePool)
    key = 'openevolve_v21'
    campaign = tmp_path / 'campaign'
    run = campaign / 'runs' / 'r'
    (run / 'candidates' / 'child').mkdir(parents=True)
    (run / 'candidates' / 'child' / 'train.py').write_text('x=2\n')
    point = dict(proposal=1, candidate_id='child', parent_ids=['unavailable'], valid=True)
    data = dict(available=True, campaign=str(campaign), runs=[dict(run_id='r', points=[point])])
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps(dict(campaigns={key: data})))
    audit = service.process_snapshot(snapshot, tmp_path / 'reviews', tmp_path / 'state', 'test', 1)
    row = json.loads((tmp_path / 'reviews' / (key + '.json')).read_text())['rows'][0]
    assert row['classification'] == 'parent_source_unavailable'
    assert row['parent_source_unavailable_provenance']['parent_ids'] == ['unavailable']
    assert not row['fingerprint_complete'] and not any(row['fingerprint'].values())
    assert json.loads((tmp_path / 'state' / 'queue' / (key + '.json')).read_text()) == []
    assert audit['fully_classified']


def test_queue_parent_sha_selects_an_exact_retained_source_over_event_lineage(tmp_path, monkeypatch):
    class CapturingPool(CompletePool):
        jobs = []
        def map(self, fn, jobs, **kwargs):
            type(self).jobs = list(jobs)
            return super().map(fn, type(self).jobs, **kwargs)

    monkeypatch.setattr(service, 'ProcessPoolExecutor', CapturingPool)
    key = 'openevolve_v21'
    campaign = tmp_path / 'campaign'
    run = campaign / 'runs' / 'r'
    sources = {'lineage-parent': 'x=1\n', 'queue-parent': 'x=2\n', 'child': 'x=3\n'}
    for candidate, code in sources.items():
        directory = run / 'candidates' / candidate
        directory.mkdir(parents=True)
        (directory / 'train.py').write_text(code)
    child = {'train.py': sources['child']}
    queue_parent = {'train.py': sources['queue-parent']}
    point = dict(proposal=1, candidate_id='child', parent_ids=['lineage-parent'], valid=True)
    data = dict(available=True, campaign=str(campaign), runs=[dict(run_id='r', points=[point])])
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps(dict(campaigns={key: data}, review_queue_provenance={key: [
        dict(run_id='r', proposal=1, candidate_id='child', source_sha256=service.sha(child),
             parent_source_sha256=[service.sha(queue_parent)])]})))
    service.process_snapshot(snapshot, tmp_path / 'reviews', tmp_path / 'state', 'test', 1)
    row = json.loads((tmp_path / 'reviews' / (key + '.json')).read_text())['rows'][0]
    assert row['parent_source_sha256'] == [service.sha(queue_parent)]
    assert row['parent_reviews'][0]['source_sha256'] == service.sha(queue_parent)
    assert CapturingPool.jobs[0][1] == queue_parent
    assert CapturingPool.jobs[0][1] != {'train.py': sources['lineage-parent']}


def test_unavailable_queue_parent_sha_is_terminal_and_never_falls_back_to_event_lineage(tmp_path, monkeypatch):
    class CapturingPool(InlinePool):
        jobs = []
        def map(self, fn, jobs, **kwargs):
            type(self).jobs = list(jobs)
            return super().map(fn, type(self).jobs, **kwargs)

    monkeypatch.setattr(service, 'ProcessPoolExecutor', CapturingPool)
    key = 'openevolve_v21'
    campaign = tmp_path / 'campaign'
    run = campaign / 'runs' / 'r'
    for candidate, code in [('lineage-parent', 'x=1\n'), ('child', 'x=3\n')]:
        directory = run / 'candidates' / candidate
        directory.mkdir(parents=True)
        (directory / 'train.py').write_text(code)
    child = {'train.py': 'x=3\n'}
    unavailable = 'queue-parent-sha-not-retained'
    point = dict(proposal=1, candidate_id='child', parent_ids=['lineage-parent'], valid=True)
    data = dict(available=True, campaign=str(campaign), runs=[dict(run_id='r', points=[point])])
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps(dict(campaigns={key: data}, review_queue_provenance={key: [
        dict(run_id='r', proposal=1, candidate_id='child', source_sha256=service.sha(child),
             parent_source_sha256=[unavailable])]})))
    audit = service.process_snapshot(snapshot, tmp_path / 'reviews', tmp_path / 'state', 'test', 1)
    row = json.loads((tmp_path / 'reviews' / (key + '.json')).read_text())['rows'][0]
    proof = row['parent_source_unavailable_provenance']
    assert row['classification'] == 'parent_source_unavailable'
    assert row['parent_source_sha256'] == [unavailable]
    assert proof['kind'] == 'queue_recorded_parent_source_unavailable'
    assert proof['parent_source_sha256'] == [unavailable]
    assert CapturingPool.jobs == []
    assert audit['fully_classified']


def test_queue_provenance_is_captured_into_the_worker_snapshot(tmp_path):
    state = tmp_path / 'state'
    (state / 'queue').mkdir(parents=True)
    rows = [dict(run_id='r', proposal=1, candidate_id='child', source_sha256='child-sha',
                 parent_source_sha256=['parent-sha'])]
    (state / 'queue' / 'openevolve_v21.json').write_text(json.dumps(rows))
    snapshot = {'campaigns': {'openevolve_v21': {'available': True}}}
    captured = service.queue_provenance_for_snapshot(snapshot, state)
    assert captured['review_queue_provenance'] == {'openevolve_v21': rows}
    assert 'review_queue_provenance' not in snapshot


def test_paths_cannot_escape_candidate_directory(tmp_path):
    run = tmp_path / 'r'
    run.mkdir()
    with pytest.raises(ValueError):
        service.safe_candidate(run, '../outside')
    with pytest.raises(ValueError):
        service.safe_candidate(run, 'candidates/..')
    assert service.safe_candidate(run, 'candidates/ok') == run / 'candidates' / 'ok'


def test_complete_fingerprint_cannot_be_invented_by_a_flag():
    row = dict(source_sha256='source')
    with pytest.raises(ValueError, match='missing components'):
        service.apply_result(row, dict(classification='preserving', notes='Known ordinary edit',
                                      evidence=['source'], fingerprint={'mixing': 'GRU'},
                                      fingerprint_complete=True), 'openevolve_v21_tiny_kws_rnn', 'test')


def test_complete_candidate_profile_does_not_replace_a_supported_transition():
    key = 'openevolve_v21'
    row = dict(source_sha256='source', classification='changing', notes='Prior source witness',
               adjudication_code_sha256='prior-engine', component_evidence=['prior witness'])
    result = dict(classification='uncertain', notes='Parent profile pending', evidence=['candidate source'],
                  fingerprint=seed_fingerprint(key), fingerprint_complete=True)
    service.apply_result(row, result, key, 'new-engine')
    assert row['classification'] == 'changing'
    assert row['notes'] == 'Prior source witness'
    assert row['adjudication_code_sha256'] == 'prior-engine'
    assert row['fingerprint_complete']
    assert row['fingerprint_provenance']['engine'] == 'new-engine'
    assert row['fingerprint_provenance']['evidence'] == ['candidate source']


def test_attempted_architecture_keeps_source_supported_execution_failure():
    row = dict(source_sha256='source')
    result = dict(classification='changing', notes='Fast/slow recurrence introduced',
                  changed_components=['state'], evidence=['Declared gate expects93 channels; caller supplies97'],
                  fingerprint=seed_fingerprint('openevolve_v21_tiny_kws_rnn'), fingerprint_complete=True,
                  implemented=True, executable=False, execution_diagnostic='97 vs93 channels at slow gate')
    service.apply_result(row, result, 'openevolve_v21_tiny_kws_rnn', 'engine')
    assert row['classification'] == 'changing' and row['fingerprint_complete']
    assert row['implemented'] is True and row['executable'] is False
    assert row['execution_diagnostic'] == result['execution_diagnostic']


def undefined_constructor_fixture():
    source = {'train.py': 'class Model:\n    def __init__(self):\n        self.layer = Missing(3, 4)\n'}
    result = dict(classification='invalid_source', invalid_kind='undefined_constructor',
                  implemented=False, executable=False, fingerprint={}, fingerprint_complete=False,
                  missing_symbols=['Missing'], notes='Missing model constructor has no definition.', evidence=[
                      dict(kind='SHA-256-bound direct review', after_source_sha256=service.sha(source)),
                      dict(kind='undefined reached constructor', file='train.py', scope='Model.__init__',
                           line=3, symbol='Missing', code='Missing(3, 4)')])
    return source, result


def test_undefined_constructor_requires_exact_source_call_and_absent_binding():
    source, result = undefined_constructor_fixture()
    proof = service.undefined_constructor_proof(result, source)
    assert proof['missing_symbols'] == ['Missing']
    for addition in ['\nMissing = object\n', '\nfrom helper import Missing\n', '\nfrom helper import *\n', '\nexec(code)\n']:
        changed = {'train.py': source['train.py'] + addition}
        result['evidence'][0]['after_source_sha256'] = service.sha(changed)
        with pytest.raises(ValueError):
            service.undefined_constructor_proof(result, changed)
    source, result = undefined_constructor_fixture()
    result['evidence'][1]['line'] = 4
    with pytest.raises(ValueError, match='claimed call'):
        service.undefined_constructor_proof(result, source)
    result['evidence'][0]['after_source_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='source binding'):
        service.undefined_constructor_proof(result, source)


@pytest.mark.parametrize('point, admitted', [
    ({'valid': False, 'failure_kind': 'execution'}, True),
    ({'valid': True, 'failure_kind': 'execution'}, False),
    ({'valid': False, 'failure_kind': 'nonqualification'}, False),
    ({'failure_kind': 'execution'}, False),
])
def test_missing_constructor_needs_independent_campaign_failure(point, admitted):
    source, result = undefined_constructor_fixture()
    result['invalid_source_provenance'] = service.undefined_constructor_proof(result, source)
    row = dict(source_sha256=service.sha(source), fingerprint_complete=True, fingerprint={'mixing': 'old'},
               family='old', family_signature={'old': 'family'})
    assert service.apply_result(row, result, 'tiny_adderboard_v21', 'engine', point) is admitted
    assert row['classification'] == ('invalid_source' if admitted else 'uncertain')
    assert row['fingerprint_complete'] is False
    if admitted:
        assert row['fingerprint'] == {} and row['family'] is None and 'family_signature' not in row
        assert row['implemented'] is False and row['executable'] is False
        row.update(run_id='r', proposal=1, candidate_id='bad', training={}, inference={})
        point = point | dict(proposal=1, candidate_id='bad')
        doc = dict(schema_version='1.0', campaign='campaign', rows=[row])
        campaign = dict(campaign='campaign', runs=[dict(run_id='r', points=[point])])
        summary, residuals = service.validate(doc, campaign, 'addition')
        assert summary['fully_classified'] and not residuals
        row['invalid_source_provenance']['source_sha256'] = 'wrong'
        with pytest.raises(ValueError, match='independent campaign failure'):
            service.validate(doc, campaign, 'addition')


def test_one_reviewer_bug_is_cached_for_review_instead_of_aborting_the_campaign(tmp_path, monkeypatch):
    class Reviewer:
        @staticmethod
        def review_pair(*args):
            raise IndexError('empty recurrent graph')
    monkeypatch.setattr(service.importlib, 'import_module', lambda name: Reviewer)
    cache = tmp_path / 'comparison.json'
    job = ('uci_har_pareto_v21', {'train.py': 'a=1'}, {'train.py': 'a=2'}, str(cache), 'engine', 'processor')
    result = service.review_job(job)
    assert result['review']['error'] == 'IndexError: empty recurrent graph'
    assert 'review_pair' in result['review']['error_traceback']
    assert service.review_job(job) == result


def test_conflicting_candidate_profiles_cannot_reuse_a_previously_complete_review(tmp_path, monkeypatch):
    key = 'openevolve_v21'
    class ConflictingPool(InlinePool):
        def map(self, fn, jobs, **kwargs):
            answers = []
            for index, job in enumerate(jobs):
                fingerprint = seed_fingerprint(key)
                fingerprint['mixing'] = 'conflicting description' + str(index)
                answers.append(dict(review=dict(classification='preserving', notes='source review',
                                                evidence=['source'], fingerprint=fingerprint,
                                                fingerprint_complete=True)))
            return answers
    monkeypatch.setattr(service, 'ProcessPoolExecutor', ConflictingPool)
    campaign = tmp_path / 'campaign'
    run = campaign / 'runs' / 'r'
    for candidate, code in [('parent_a', 'a=1\n'), ('parent_b', 'a=2\n'), ('child', 'a=3\n')]:
        directory = run / 'candidates' / candidate
        directory.mkdir(parents=True)
        (directory / 'train.py').write_text(code)
    point = dict(proposal=1, candidate_id='child', parent_ids=['parent_a', 'parent_b'])
    data = dict(available=True, campaign=str(campaign), runs=[dict(run_id='r', points=[point])])
    output = tmp_path / 'reviews'
    output.mkdir()
    row = service.blank_row(data['runs'][0], point, {'train.py': 'a=3\n'}, 'old')
    row.update(classification='preserving', fingerprint=seed_fingerprint(key), fingerprint_complete=True,
               fingerprint_provenance={'kind': 'prior review'}, family='prior family')
    (output / (key + '.json')).write_text(json.dumps(dict(rows=[row])))
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps(dict(campaigns={key: data})))
    audit = service.process_snapshot(snapshot, output, tmp_path / 'state', 'new', 1)
    assert not audit['fully_classified']
    queue = json.loads((tmp_path / 'state' / 'queue' / (key + '.json')).read_text())
    assert queue[0]['classification'] == 'uncertain' and not queue[0]['fingerprint_complete']
    assert 'disagree' in queue[0]['error']


def test_empty_inventory_cannot_claim_completion(tmp_path):
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text('{"campaigns":{}}')
    audit = service.process_snapshot(snapshot, tmp_path / 'reviews', tmp_path / 'state', 'test', 1)
    assert not audit['fully_classified']


class CompletePool(InlinePool):
    def map(self, fn, jobs, **kwargs):
        return [dict(review=dict(classification='preserving', notes='complete source proof', evidence=['source'],
                                 fingerprint=seed_fingerprint(job[0]), fingerprint_complete=True)) for job in jobs]


def two_campaigns(tmp_path):
    campaigns = {}
    for key in ['openevolve_v21_nanogpt', 'openevolve_v21']:
        campaign = tmp_path / key
        source = campaign / 'runs/r/candidates/seed'
        source.mkdir(parents=True)
        (source / 'train.py').write_text('x=1\n')
        campaigns[key] = dict(available=True, campaign=str(campaign), runs=[dict(run_id='r', points=[
            dict(proposal=0, candidate_id='seed', is_seed=True, valid=True, parent_ids=[])])])
    return campaigns


def test_validated_campaign_survives_later_campaign_failure_without_false_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'ProcessPoolExecutor', CompletePool)
    campaigns = two_campaigns(tmp_path)
    campaigns['openevolve_v21']['runs'][0]['run_id'] = '../escape'
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps(dict(campaigns=campaigns)))
    with pytest.raises(ValueError, match='Run identity escapes'):
        service.process_snapshot(snapshot, tmp_path / 'reviews', tmp_path / 'state', 'test', 1)
    assert (tmp_path / 'reviews/openevolve_v21_nanogpt.json').exists()
    audit = json.loads((tmp_path / 'state/audit.json').read_text())
    assert not audit['fully_classified']
    assert audit['pending_campaigns'] == ['openevolve_v21']


@pytest.mark.parametrize('revoke_before', [False, True])
def test_processor_revocation_withholds_only_affected_campaign(tmp_path, monkeypatch, revoke_before):
    state = tmp_path / 'state'
    state.mkdir()
    def revoke():
        service.atomic_json(state / 'revoked-processors.json', {'bad': 'source-binding correction pending'})
    if revoke_before:
        revoke()
    class RevokingPool(CompletePool):
        def map(self, fn, jobs, **kwargs):
            result = super().map(fn, jobs, **kwargs)
            revoke()
            return result
    monkeypatch.setattr(service, 'ProcessPoolExecutor', RevokingPool)
    monkeypatch.setattr(service, 'processor_version', lambda key: 'bad' if key == 'openevolve_v21_nanogpt' else 'good')
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps(dict(campaigns=two_campaigns(tmp_path))))
    audit = service.process_snapshot(snapshot, tmp_path / 'reviews', state, 'test', 1)
    assert not (tmp_path / 'reviews/openevolve_v21_nanogpt.json').exists()
    assert (tmp_path / 'reviews/openevolve_v21.json').exists()
    assert not audit['fully_classified']
    assert audit['pending_campaigns'] == ['openevolve_v21_nanogpt']
    assert audit['pending_inventory'][0]['processor_sha256'] == 'bad'
