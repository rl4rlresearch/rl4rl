"""Validate snapshot-bound review coverage and publish the diagnostic atomically.

Processing every row is not the same as resolving every row. The audit retains
both counts, and --require-complete fails if any transition or fingerprint is
unresolved. No completion flag is inferred from a successful extraction pass.
"""
import argparse
import collections
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from experiments.review_ontology_sources import CORE, TASK_KEYS, task_for
from experiments.review_ontology_v2 import atomic_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(doc, campaign, task):
    if doc.get('schema_version') != '1.0' or doc.get('campaign') != campaign['campaign']:
        raise ValueError('Review schema/campaign mismatch')
    expected = {(run['run_id'], p['proposal']): p for run in campaign['runs']
                for p in run['points'] if p.get('candidate_id')}
    seen = set()
    components = set(CORE + TASK_KEYS[task].split())
    counts = collections.Counter()
    complete = 0
    residuals = []
    for row in doc['rows']:
        key = (row['run_id'], row['proposal'])
        point = expected.get(key)
        if key in seen or not point or point['candidate_id'] != row['candidate_id']:
            raise ValueError('Duplicate, stale or unknown review: ' + str(key))
        seen.add(key)
        classification = row['classification']
        if classification not in {'changing', 'preserving', 'mixed', 'uncertain', 'unannotated', 'invalid_source',
                                  'source_unavailable', 'parent_source_unavailable'}:
            raise ValueError('Unknown classification: ' + str(key))
        for group in ('fingerprint', 'training', 'inference'):
            values = row.get(group)
            if not isinstance(values, dict) or any(v is not None and (not isinstance(v, str) or not v.strip()) for v in values.values()):
                raise ValueError('Invalid categorical values: ' + str(key))
        if row.get('fingerprint_complete'):
            if not components <= row['fingerprint'].keys() or any(not row['fingerprint'][c] for c in components):
                raise ValueError('Partial fingerprint claims completeness: ' + str(key))
            if not row.get('fingerprint_provenance'):
                raise ValueError('Complete fingerprint lacks provenance: ' + str(key))
            complete += 1
        if classification == 'source_unavailable':
            proof = row.get('source_unavailable_provenance', {})
            if (proof.get('kind') != 'candidate_source_unavailable' or
                    proof.get('candidate_id') != point['candidate_id'] or
                    not isinstance(proof.get('artifact_path'), str) or
                    not isinstance(proof.get('error'), str) or not proof['error'].strip() or
                    row.get('fingerprint_complete') or row.get('family') is not None):
                raise ValueError('Source-unavailable status lacks retained artifact evidence: ' + str(key))
        elif classification == 'parent_source_unavailable':
            proof = row.get('parent_source_unavailable_provenance', {})
            missing = proof.get('parent_ids')
            paths = proof.get('parent_artifact_paths')
            recorded_event_parent = (proof.get('kind') == 'recorded_parent_source_unavailable' and
                                     proof.get('candidate_id') == point['candidate_id'] and
                                     isinstance(missing, list) and missing and
                                     not any(parent not in point.get('parent_ids', []) for parent in missing) and
                                     isinstance(paths, list) and paths == ['candidates/' + parent for parent in missing])
            recorded_queue_parent = (proof.get('kind') == 'queue_recorded_parent_source_unavailable' and
                                     proof.get('candidate_id') == point['candidate_id'] and
                                     proof.get('run_id') == row['run_id'] and proof.get('proposal') == row['proposal'] and
                                     proof.get('source_sha256') == row.get('source_sha256') and
                                     isinstance(proof.get('parent_source_sha256'), list) and
                                     bool(proof['parent_source_sha256']) and
                                     all(isinstance(value, str) and value for value in proof['parent_source_sha256']) and
                                     proof['parent_source_sha256'] == row.get('parent_source_sha256'))
            if (not (recorded_event_parent or recorded_queue_parent) or
                    row.get('fingerprint_complete') or row.get('family') is not None):
                raise ValueError('Parent-source-unavailable status lacks retained parent evidence: ' + str(key))
        elif classification == 'invalid_source':
            proof = row.get('invalid_source_provenance', {})
            undefined = (row.get('invalid_kind') == 'undefined_constructor' and
                         proof.get('kind') == 'undefined_constructor' and proof.get('missing_symbols') and
                         row.get('source_sha256') and proof.get('source_sha256') == row['source_sha256'] and
                         row.get('adjudication_code_sha256') and row.get('component_evidence') and
                         point.get('failure_kind') == 'execution' and point.get('valid') is False)
            syntax = row.get('invalid_kind') in {None, 'syntax_error'} and point.get('failure_kind') == 'source_preflight'
            if (not (undefined or syntax) or point.get('valid') or row.get('implemented') is not False or
                    row.get('executable') is not False or row.get('fingerprint_complete')):
                raise ValueError('Invalid-source label lacks source and independent campaign failure support: ' + str(key))
        elif not row.get('source_sha256') or not row.get('adjudication_code_sha256'):
            raise ValueError('Review lacks source/engine binding: ' + str(key))
        counts[classification] += 1
        if (classification in {'uncertain', 'unannotated'} or
                not row.get('fingerprint_complete') and classification not in {
                    'invalid_source', 'source_unavailable', 'parent_source_unavailable'}):
            residuals.append({k: row.get(k) for k in ('run_id', 'proposal', 'candidate_id', 'classification', 'source_sha256', 'notes')}
                             | {'unresolved_components': sorted(c for c in components if row['fingerprint'].get(c) is None)})
    if seen != expected.keys():
        raise ValueError('Review does not cover the complete captured snapshot')
    return {'rows': len(seen), 'classifications': dict(counts),
            'resolved_transitions': len(seen) - counts['uncertain'] - counts['unannotated'],
            'complete_fingerprints': complete,
            'fully_classified': not residuals}, residuals


def publish(snapshot_path, review_dir, output, engine=None, require_complete=False, write=False):
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report = {'created_at': stamp, 'snapshot_sha256': digest(snapshot_path),
              'snapshot_generated_at': snapshot.get('generated_at'),
              'campaigns': {}, 'published': False}
    expected_engine = None
    if engine:
        manifest = json.loads((engine / 'manifest.json').read_text(encoding='utf-8'))
        for name, sha in manifest['files'].items():
            if digest(engine / 'experiments' / name) != sha:
                raise ValueError('Frozen engine changed: ' + name)
        report['engine'] = manifest
        expected_engine = hashlib.sha256(b''.join((engine / 'experiments' / name).read_bytes() for name in (
            'ontology_semantics.py', 'ontology_transition_review.py', 'ontology_affine_readout.py',
            'ontology_seed_rubric.py', 'adjudicate_ontology_v2.py'))).hexdigest()[:12]
        launcher = engine / 'run_cached_adjudication.py'
        if launcher.exists():
            report['memoization_launcher_sha256'] = digest(launcher)
    docs, queues = {}, {}
    for key, campaign in snapshot['campaigns'].items():
        if not campaign.get('available'):
            continue
        file = review_dir / (key + '.json')
        doc = json.loads(file.read_text(encoding='utf-8'))
        if expected_engine and (doc.get('adjudication', {}).get('code_sha256') != expected_engine or
                                any(r.get('adjudication_code_sha256') != expected_engine for r in doc['rows'])):
            raise ValueError('Review was not produced by the selected frozen engine: ' + key)
        summary, queue = validate(doc, campaign, task_for(key))
        summary['review_sha256'] = digest(file)
        if doc.get('semantic_references'):
            reference_file = Path(__file__).with_name('ontology_reference_reviews.json')
            if digest(reference_file) != doc['semantic_references']['sha256']:
                raise ValueError('Source-specific semantic references changed: ' + key)
            summary['semantic_references'] = doc['semantic_references']
        base = review_dir.parent / (key + '.json')
        if base.exists():
            summary['extraction_input_sha256'] = digest(base)
        report['campaigns'][key] = summary
        docs[key], queues[key] = doc, queue
    report['fully_classified'] = all(c['fully_classified'] for c in report['campaigns'].values())
    atomic_json(review_dir / 'audit.json', report)
    for key, queue in queues.items():
        atomic_json(review_dir / (key + '-review-queue.json'), queue)
    if require_complete and not report['fully_classified']:
        raise ValueError('Review remains incomplete; see audit.json and campaign review queues')
    if write:
        backup = output / 'backups' / stamp
        for key, doc in docs.items():
            destination = output / (key + '.json')
            if destination.exists():
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, backup / destination.name)
            atomic_json(destination, doc)
        report['published'] = True
        report['backup'] = str(backup)
        atomic_json(review_dir / 'audit.json', report)
        atomic_json(output / 'review-audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, default=Path('outputs/ontology/v2/dashboard-snapshot.json'))
    parser.add_argument('--reviews', type=Path, default=Path('outputs/ontology/v2/final'))
    parser.add_argument('--output', type=Path, default=Path('outputs/ontology'))
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    print(json.dumps(publish(args.snapshot, args.reviews, args.output, args.engine, args.require_complete, args.publish), indent=2))
