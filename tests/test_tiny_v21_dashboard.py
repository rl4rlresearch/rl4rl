from pathlib import Path

from experiments import live_trajectory_dashboard as dashboard
from experiments.c0c3_factorial import file_lock


def test_cpu_ledger_is_not_priced_as_gpu(tmp_path):
    (tmp_path / 'modal-usage.jsonl').write_text(
        '{"run_id":"tiny","opportunity":1,"gpu_name":"CPU (2 cores, 4 GiB)",'
        '"worker_seconds":30,"status":"completed"}\n'
    )
    indexed, summary = dashboard.modal_usage_index(tmp_path, h100_price_per_second=0.001)
    assert indexed == {}
    assert summary['gpu_cost'] == 0
    assert summary['cpu_worker_seconds'] == 30
    assert not summary['available']


def test_windows_status_requires_live_lock_not_saved_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, 'WINDOWS_VIEW_ONLY', True)
    run = tmp_path / 'runs/tiny-live'
    run.mkdir(parents=True)
    lock = run / '.trajectory-controller.lock'
    lock.write_text('{"pid":1234}')
    assert dashboard.runtime_activity_snapshot([tmp_path])['run_controllers'] == set()
    with lock.open('r+') as handle:
        file_lock.flock(handle.fileno(), file_lock.LOCK_EX | file_lock.LOCK_NB)
        try:
            activity = dashboard.runtime_activity_snapshot([tmp_path])
            assert activity['run_controllers'] == {'tiny-live'}
            stage = dashboard.operational_stage(
                run, {'run_id':'tiny-live', 'active':{'index':1}}, status='running',
                desired_state=None, campaign_desired=None, evaluator_backend='hybrid_modal',
                runtime_activity=activity, campaign=tmp_path, campaign_subject_limit=None,
                campaign_active_opportunities=None, semantic=False,
            )
            assert stage['kind'] == 'active'
        finally:
            file_lock.flock(handle.fileno(), file_lock.LOCK_UN)
    assert not dashboard.runtime_activity_snapshot([tmp_path])['run_controllers']


def test_new_campaign_registered_in_browser():
    assert dashboard.DEFAULT_TINY_ADDERBOARD_V21.name == 'tiny-v21'
    html = (Path(dashboard.__file__).with_suffix('.html')).read_text(encoding='utf-8')
    assert "'tiny_adderboard_v21'" in html
