"""Continuous source-bound ontology review with immutable reviewer versions.

Unrecognized mechanisms enter a durable semantic-review queue. Processing a
proposal never implies that it was resolved. Candidate Python is read, not run.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from copy import deepcopy
import hashlib
import importlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from experiments.publish_ontology_reviews import validate
from experiments.review_ontology_sources import CORE, TASK_KEYS, task_for
from experiments.review_ontology_v2 import atomic_json

PROCESSORS = {
    'openevolve_v21': 'ontology_review_addition',
    'tiny_adderboard_v21': 'ontology_review_addition',
    'uci_har_pareto_v21': 'ontology_review_recurrent',
    'openevolve_v21_tiny_kws_rnn': 'ontology_review_recurrent',
    'openevolve_v21_fashion_mnist': 'ontology_review_vision_lm',
    'openevolve_v21_nanogpt': 'ontology_review_vision_lm',
}
# Tiny AdderBoard is deliberately outside the current post-campaign study.  Keep
# its existing artifact immutable, but never spend reviewer work or completion
# gates on it.
EXCLUDED_CAMPAIGNS = {'tiny_adderboard_v21'}
RESOLVED = {'preserving', 'changing', 'mixed', 'invalid_source', 'source_unavailable',
            'parent_source_unavailable'}
CAMPAIGN_ORDER = {'openevolve_v21_nanogpt': 0, 'openevolve_v21_fashion_mnist': 1}


class ReviewBusy(RuntimeError):
    pass


def canonicalize_exact_source_fingerprints(rows):
    """Replace a stale cached profile with the one direct review of that source.

    A transition classification remains specific to its parent, but an ontology
    fingerprint describes the candidate program itself.  If the same candidate
    source occurs at more than one proposal point, a direct parent-child review
    is therefore authoritative over an older cached profile for that exact
    source.  Two direct reviews that disagree remain a real conflict and are
    left for the existing uncertainty gate below.
    """
    direct = {}
    disputed = set()
    for row in rows:
        if not row.get('fingerprint_complete') or not row.get('parent_reviews'):
            continue
        source_hash = row['source_sha256']
        fingerprint = row.get('fingerprint', {})
        prior = direct.get(source_hash)
        if prior is None:
            direct[source_hash] = (fingerprint, row['candidate_id'])
        elif prior[0] != fingerprint:
            disputed.add(source_hash)
    for source_hash in disputed:
        direct.pop(source_hash, None)
    for row in rows:
        canonical = direct.get(row['source_sha256'])
        if canonical is None or row.get('fingerprint') == canonical[0]:
            continue
        row.update(
            fingerprint=deepcopy(canonical[0]),
            fingerprint_complete=True,
            fingerprint_provenance={
                'kind': 'exact reviewed source match',
                'source_sha256': row['source_sha256'],
                'anchor_candidate_id': canonical[1],
            },
        )


def acquire_lock(state):
    """OS-held lock: a stale status file never blocks recovery after a crash."""
    handle = (state / 'review.lock').open('a+b')
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise ReviewBusy('Another ontology review holds the process lock') from exc
    return handle


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


@lru_cache(maxsize=None)
def processor_version(key):
    """Hash the frozen task reviewer, local imports and reference data it reads.

    An unrelated task's rule edit need not discard validated source comparisons.
    Dynamic task dispatch is resolved by PROCESSORS; sibling reviewer imports
    and reference filenames are explicit in the trusted reviewer source.
    """
    directory = Path(__file__).resolve().parent
    pending = [PROCESSORS[key] + '.py']
    files = {}
    while pending:
        name = pending.pop()
        if name in files:
            continue
        path = directory / name
        content = path.read_bytes()
        files[name] = hashlib.sha256(content).hexdigest()
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(content)):
            imports = []
            if isinstance(node, ast.Import):
                imports = [a.name.removeprefix('experiments.') + '.py'
                           for a in node.names if a.name.startswith('experiments.')]
            elif isinstance(node, ast.ImportFrom):
                if node.module == 'experiments':
                    imports = [a.name + '.py' for a in node.names]
                elif node.module and node.module.startswith('experiments.'):
                    imports = [node.module.removeprefix('experiments.') + '.py']
                elif node.level == 1 and node.module:
                    imports = [node.module + '.py']
            elif (isinstance(node, ast.Constant) and isinstance(node.value, str) and
                  node.value.startswith('ontology_') and node.value.endswith('.json')):
                imports = [node.value]
            pending.extend(n for n in imports if (directory / n).is_file() and n not in files)
    return sha({'files': files, 'contract': 'semantic-review-result-v3',
                'python': list(sys.version_info[:3])})


def check_revocation(state, engine):
    file = state / 'revoked-engines.json'
    if file.exists() and engine in json.loads(file.read_text(encoding='utf-8')):
        raise ValueError('Reviewer version was revoked during this pass; publication withheld')


def processor_revocation(state, processor):
    file = state / 'revoked-processors.json'
    return json.loads(file.read_text(encoding='utf-8')).get(processor) if file.exists() else None


def save_audit(audit, state, output):
    inventory_complete = bool(audit['campaigns']) and not audit['pending_inventory'] and not audit['pending_campaigns']
    audit['transitions_complete'] = inventory_complete and all(c['resolved_transitions'] == c['rows'] for c in audit['campaigns'].values())
    # A missing, identity-checked candidate artifact is terminal evidence, just
    # as an independently confirmed invalid source is.  Neither has a program
    # that could honestly receive a fingerprint, so neither should keep the
    # campaign's completion gate in the ambiguous state.
    audit['fingerprints_complete'] = inventory_complete and all(
        c['complete_fingerprints'] + c['classifications'].get('invalid_source', 0) +
        c['classifications'].get('source_unavailable', 0) +
        c['classifications'].get('parent_source_unavailable', 0) == c['rows']
        for c in audit['campaigns'].values())
    audit['fully_classified'] = audit['transitions_complete'] and audit['fingerprints_complete']
    audit['updated_at'] = utcnow()
    atomic_json(state / 'audit.json', audit)
    atomic_json(output / 'review-audit.json', audit)


def publish_campaign(key, doc, state, output, engine):
    check_revocation(state, engine)
    destination = output / (key + '.json')
    if destination.exists():
        content = destination.read_bytes()
        backup = state / 'prior-reviews' / (hashlib.sha256(content).hexdigest() + '.json')
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(content)
    atomic_json(destination, doc)


def safe_candidate(run, relative):
    run = Path(run).resolve()
    candidates = (run / 'candidates').resolve()
    candidate = (run / str(relative).replace('\\', '/')).resolve()
    if not candidates.is_relative_to(run) or candidate == candidates or not candidate.is_relative_to(candidates):
        raise ValueError('Recorded source path escapes the run candidate directory')
    return candidate


def snapshot_queue_provenance(snapshot, key):
    """Return queue rows carried by an immutable review snapshot.

    ``review_queue_provenance`` is the canonical capture name.  The aliases
    make the reader forward-compatible with a dashboard/API that exposes the
    same durable queue under a shorter name; a campaign-local capture is also
    accepted.  This function deliberately never reads the live queue: a
    worker must make decisions from its frozen snapshot only.
    """
    campaign = snapshot.get('campaigns', {}).get(key, {})
    for container in (snapshot, campaign):
        if not isinstance(container, dict):
            continue
        for field in ('review_queue_provenance', 'ontology_review_queue', 'review_queue'):
            value = container.get(field)
            if isinstance(value, dict):
                value = value.get(key) if container is snapshot else value
            if value is not None:
                if not isinstance(value, list):
                    raise ValueError('Review queue provenance must be a list for ' + key)
                return value
    return []


def queue_parent_hashes(snapshot, key, run, point, source):
    """Get the queue-bound parents for this exact candidate-source record.

    A queue row is authoritative only when all of its identity fields,
    including the child source hash, match the source observed in this pass.
    Event lineage is intentionally not considered once such a row exists.
    """
    actual = sha(source)
    matches = []
    for row in snapshot_queue_provenance(snapshot, key):
        if not isinstance(row, dict):
            continue
        if (row.get('run_id') != run['run_id'] or row.get('proposal') != point.get('proposal') or
                row.get('candidate_id') != point.get('candidate_id') or row.get('source_sha256') != actual):
            continue
        parent_hashes = row.get('parent_source_sha256')
        if isinstance(parent_hashes, str):
            parent_hashes = [parent_hashes]
        if (not isinstance(parent_hashes, list) or not parent_hashes or
                any(not isinstance(value, str) or not value for value in parent_hashes)):
            raise ValueError('Queue provenance has no parent source SHA for ' + str(point.get('candidate_id')))
        matches.append(tuple(parent_hashes))
    if not matches:
        return None
    unique = set(matches)
    if len(unique) != 1:
        raise ValueError('Conflicting queue parent source SHAs for ' + str(point.get('candidate_id')))
    return list(unique.pop())


def retained_sources_by_sha(campaign, cache):
    """Index only retained candidate directories beneath a campaign's runs root."""
    campaign = Path(campaign).resolve()
    runs = (campaign / 'runs').resolve()
    if runs.parent != campaign or not runs.is_dir():
        raise ValueError('Campaign runs directory is unavailable')
    index = {}
    for run in sorted(runs.iterdir()):
        resolved_run = run.resolve()
        if resolved_run.parent != runs or not resolved_run.is_dir():
            continue
        candidates = (resolved_run / 'candidates').resolve()
        if candidates.parent != resolved_run or not candidates.is_dir():
            continue
        for candidate in sorted(candidates.iterdir()):
            resolved_candidate = candidate.resolve()
            if resolved_candidate.parent != candidates or not resolved_candidate.is_dir():
                continue
            try:
                source = cache.read(resolved_candidate, capture=True)
            except (OSError, ValueError, UnicodeError):
                # An unreadable retained directory cannot establish an exact
                # source match and must not be used as a substitute.
                continue
            if source:
                index.setdefault(sha(source), []).append((resolved_candidate, source))
    return index


def queue_provenance_for_snapshot(snapshot, state):
    """Capture the durable queue alongside API data before a worker is frozen."""
    captured = dict(snapshot)
    existing = captured.get('review_queue_provenance')
    if existing is not None:
        if not isinstance(existing, dict):
            raise ValueError('Review queue provenance must be keyed by campaign')
        return captured
    rows_by_campaign = {}
    queue = Path(state) / 'queue'
    for key in captured.get('campaigns', {}):
        path = queue / (key + '.json')
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(rows, list):
            raise ValueError('Durable review queue is not a list: ' + str(path))
        rows_by_campaign[key] = rows
    if rows_by_campaign:
        captured['review_queue_provenance'] = rows_by_campaign
    return captured


class SourceCache:
    """Rehash source whenever the file inventory/mtime/size changes."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS sources (path TEXT PRIMARY KEY, stamp TEXT, source TEXT)')
        self.captured = {}

    def read(self, path, capture=False):
        # A captured pass observes each immutable candidate identity once, so
        # repeated parent references share exactly the same source observation.
        # A fresh SourceCache is constructed for every subsequent pass.
        if capture and path in self.captured:
            return self.captured[path]
        files = sorted(path.rglob('*.py')) if path.is_dir() else []
        if len(files) > 100:
            raise ValueError('Candidate has more than 100 Python files; source review required')
        stamps = []
        for file in files:
            if not file.resolve().is_relative_to(path.resolve()):
                raise ValueError('Source symlink leaves candidate directory')
            stat = file.stat()
            if stat.st_size > 2_000_000:
                raise ValueError('Oversized candidate source requires review')
            stamps.append((str(file.relative_to(path)).replace('\\', '/'), stat.st_mtime_ns, stat.st_size))
        stamp = sha(stamps)
        cached = self.db.execute('SELECT stamp,source FROM sources WHERE path=?', (str(path),)).fetchone()
        if cached and cached[0] == stamp:
            source = json.loads(cached[1])
        else:
            source = {name: (path / name).read_text(encoding='utf-8-sig') for name, _, _ in stamps}
            self.db.execute('INSERT OR REPLACE INTO sources VALUES (?,?,?)', (str(path), stamp, json.dumps(source)))
        if capture:
            self.captured[path] = source
        return source

    def close(self):
        self.db.commit()
        self.db.close()


def undefined_constructor_proof(result, source):
    """Validate an exact-source manual finding; never guess a missing class."""
    if (result.get('invalid_kind') != 'undefined_constructor' or
            result.get('implemented') is not False or result.get('executable') is not False or
            result.get('fingerprint_complete') or result.get('fingerprint')):
        raise ValueError('Invalid constructor finding lacks a non-implementation contract')
    evidence = result.get('evidence', [])
    if not any(e.get('kind') == 'SHA-256-bound direct review' and e.get('after_source_sha256') == sha(source)
               for e in evidence if isinstance(e, dict)):
        raise ValueError('Undefined constructor finding lacks exact source binding')
    symbols = result.get('missing_symbols')
    if not isinstance(symbols, list) or not symbols or any(not isinstance(s, str) or not s.isidentifier() for s in symbols):
        raise ValueError('Undefined constructor finding has no explicit symbols')
    trees = {name: ast.parse(text) for name, text in source.items()}
    declared = set(dir(builtins))
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                declared.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                declared.add(node.id)
            elif isinstance(node, ast.arg):
                declared.add(node.arg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if any(a.name == '*' for a in node.names):
                    raise ValueError('Wildcard import prevents missing-symbol proof')
                declared.update(a.asname or a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'exec', 'eval', 'globals', 'locals'}:
                raise ValueError('Dynamic name construction prevents missing-symbol proof')
    for symbol in symbols:
        if symbol in declared:
            raise ValueError('Claimed missing constructor has a source declaration')
        found = False
        for e in evidence:
            if not isinstance(e, dict) or e.get('kind') != 'undefined reached constructor' or e.get('symbol') != symbol:
                continue
            tree = trees.get(e.get('file'))
            if tree is None:
                continue
            owner, _, method = e.get('scope', '').partition('.')
            for cls in tree.body:
                if not isinstance(cls, ast.ClassDef) or cls.name != owner or method != '__init__':
                    continue
                for fn in cls.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == method:
                        found |= any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and
                                     n.func.id == symbol and n.lineno == e.get('line') and ast.unparse(n) == e.get('code')
                                     for n in ast.walk(fn))
        if not found:
            raise ValueError('Missing constructor evidence does not match its claimed call')
    return dict(kind='undefined_constructor', source_sha256=sha(source), missing_symbols=symbols)


def review_job(job):
    key, before, after, cache_file, engine, processor = job
    file = Path(cache_file)
    if file.exists():
        return json.loads(file.read_text(encoding='utf-8'))
    result = None
    try:
        module = importlib.import_module('experiments.' + PROCESSORS[key])
        result = module.review_pair(before, after, key)
        if result is not None:
            if not isinstance(result, dict) or result.get('classification') not in RESOLVED | {'uncertain'}:
                raise ValueError('Semantic reviewer returned an unsupported classification')
            if not result.get('notes') and not result.get('rationale'):
                raise ValueError('Semantic classification has no rationale')
            if not result.get('evidence'):
                raise ValueError('Semantic classification has no source evidence')
            if result['classification'] in {'changing', 'mixed'} and not result.get('changed_components'):
                raise ValueError('Changing classification has no component witness')
            if before == after and result['classification'] in {'changing', 'mixed'}:
                raise ValueError('Identical source cannot introduce a mechanism change')
            if result['classification'] == 'invalid_source':
                result['invalid_source_provenance'] = undefined_constructor_proof(result, after)
    except ModuleNotFoundError as exc:
        if exc.name != 'experiments.' + PROCESSORS[key]:
            raise
    except Exception as exc:
        # A malformed/unhandled source graph is one queued review, not a reason
        # to discard every other campaign's completed work or retry forever.
        result = {'error': type(exc).__name__ + ': ' + str(exc),
                  'error_traceback': traceback.format_exc(limit=8)}
    wrapped = {'engine': engine, 'processor_sha256': processor,
               'before_sha256': sha(before), 'after_sha256': sha(after), 'review': result}
    atomic_json(file, wrapped)
    return wrapped


def blank_row(run, point, source, engine):
    return dict(run_id=run['run_id'], proposal=point['proposal'], candidate_id=point['candidate_id'],
                classification='uncertain', family=None, fingerprint={}, training={}, inference={}, settings={},
                proposed_change=None, implemented=None, executable=None, source_sha256=sha(source),
                adjudication_code_sha256=engine, reviewer='Task-specific semantic review',
                notes='Pending source-based semantic review.')


def apply_result(row, result, key, engine, point=None):
    if not result or result.get('error'):
        return False
    if result.get('classification') == 'invalid_source':
        proof = result.get('invalid_source_provenance', {})
        if (not point or point.get('valid') is not False or point.get('failure_kind') != 'execution' or
                proof.get('kind') != 'undefined_constructor' or proof.get('source_sha256') != row['source_sha256']):
            row.update(classification='uncertain', fingerprint_complete=False, family=None,
                       notes='Missing constructor finding requires matching campaign execution-failure evidence.')
            return False
        row.update(classification='invalid_source', invalid_kind='undefined_constructor',
                   implemented=False, executable=False, fingerprint={}, fingerprint_complete=False, family=None,
                   notes=result['notes'], component_evidence=result['evidence'],
                   invalid_source_provenance=proof, missing_symbols=proof['missing_symbols'],
                   adjudication_code_sha256=engine, reviewer='Task-specific semantic review')
        row.pop('family_signature', None)
        row.pop('fingerprint_provenance', None)
        return True
    fingerprint = result.get('fingerprint', {})
    required = CORE + TASK_KEYS[task_for(key)].split()
    complete = result.get('fingerprint_complete', all(isinstance(fingerprint.get(k), str) and fingerprint[k].strip() for k in required))
    if complete and any(not isinstance(fingerprint.get(k), str) or not fingerprint[k].strip() for k in required):
        raise ValueError('Reviewer claimed a complete fingerprint with missing components')
    resolved = result['classification'] in {'preserving', 'changing', 'mixed'}
    if resolved or row.get('classification') not in RESOLVED:
        row.update(classification=result['classification'], notes=result.get('notes') or result['rationale'],
                   changed_components=result.get('changed_components', []), component_evidence=result['evidence'],
                   adjudication_code_sha256=engine, reviewer='Task-specific semantic review',
                   review_method='Source-audited task-specific mechanism rules')
    if complete or not row.get('fingerprint_complete'):
        row.update(fingerprint=fingerprint, fingerprint_complete=bool(complete))
        if complete:
            row['fingerprint_provenance'] = {'kind': 'source-traced task-specific semantic review',
                                           'engine': engine, 'source_sha256': row['source_sha256'],
                                           'evidence': result['evidence']}
    for group in ('training', 'inference', 'settings'):
        if isinstance(result.get(group), dict):
            row[group] = result[group]
    for field in ('implemented', 'executable'):
        if type(result.get(field)) is bool:
            row[field] = result[field]
    if isinstance(result.get('execution_diagnostic'), str):
        row['execution_diagnostic'] = result['execution_diagnostic']
    signature = result.get('family_signature')
    if signature:
        row['family_signature'] = signature
        label = result.get('family_label') or fingerprint.get('mixing') or 'reviewed mechanism'
        row['family'] = str(label)[:100] + ' · ' + sha(signature)[:10]
    return True


def reconcile_families(rows):
    """Use reviewed equivalence edges to reconcile different semantic encodings.

    Source identities bind the evidence, but only a semantic signature defines
    the displayed family ID. Equivalence never supplies missing FP components.
    """
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    usable = [r for r in rows if r.get('source_sha256') != sha({}) and
              r.get('classification') not in {'invalid_source', 'source_unavailable', 'parent_source_unavailable'}]
    signatures = {}
    for row in usable:
        source = row['source_sha256']
        find(source)
        signature = row.get('family_signature')
        if row.get('fingerprint_complete') and signature:
            identity = sha(signature)
            if identity in signatures:
                union(source, signatures[identity])
            signatures[identity] = source
        edges = row.get('parent_reviews', [])
        if not edges and row.get('classification') == 'preserving':
            edges = [{'source_sha256': p, 'classification': 'preserving'} for p in row.get('parent_source_sha256', [])]
        for edge in edges:
            previous = edge.get('source_sha256')
            if previous and previous != sha({}) and edge.get('classification') == 'preserving':
                union(source, previous)
    anchors = {}
    for row in usable:
        if row.get('fingerprint_complete') and row.get('family_signature'):
            anchors.setdefault(find(row['source_sha256']), []).append(row)
    canonical = {}
    for group, members in anchors.items():
        reference = min(members, key=lambda r: (sha(r['family_signature']), str(r.get('family') or '')))
        signature_id = sha(reference['family_signature'])[:10]
        label = str(reference.get('family') or reference.get('fingerprint', {}).get('mixing') or 'reviewed mechanism').split(' · ')[0][:100]
        canonical[group] = (label + ' · ' + signature_id, signature_id)
    conflicts = []
    for row in usable:
        group = find(row['source_sha256'])
        if group in canonical:
            row['family'], row['family_id'] = canonical[group]
            row['family_provenance'] = {'kind': 'semantic signature and source-supported preserving equivalence',
                                        'canonical_signature_id': canonical[group][1]}
        edges = row.get('parent_reviews', [])
        if not edges and row.get('classification') in {'changing', 'mixed'} and len(row.get('parent_source_sha256', [])) == 1:
            edges = [{'source_sha256': row['parent_source_sha256'][0], 'classification': row['classification']}]
        contradictions = [e['source_sha256'] for e in edges if e.get('source_sha256') and
                          e.get('classification') in {'changing', 'mixed'} and find(e['source_sha256']) == group]
        if contradictions:
            row['classification_before_consistency_check'] = row['classification']
            row.update(classification='uncertain', semantic_review_error='Changing witness contradicts the reviewed family-equivalence graph',
                       notes='Family consistency review required: a changing parent comparison connects source-equivalent or preserving-equivalent architectures.')
            conflicts.append({'source_sha256': row['source_sha256'], 'parent_source_sha256': contradictions})
    return conflicts


def process_snapshot(snapshot_path, output, state, engine, workers=2):
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    state.mkdir(parents=True, exist_ok=True)
    cache = SourceCache(state / 'source-index.sqlite')
    audit = {'generated_at': utcnow(), 'snapshot_generated_at': snapshot.get('generated_at'),
             'snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
             'engine': engine, 'campaigns': {}, 'pending_inventory': [],
             'pending_campaigns': [key for key in snapshot['campaigns']
                                   if key in PROCESSORS and key not in EXCLUDED_CAMPAIGNS],
             'excluded_campaigns': sorted(key for key in snapshot['campaigns']
                                         if key in EXCLUDED_CAMPAIGNS)}
    save_audit(audit, state, output)
    try:
        for key, campaign in sorted(snapshot['campaigns'].items(), key=lambda item: CAMPAIGN_ORDER.get(item[0], 2)):
            if key not in PROCESSORS or key in EXCLUDED_CAMPAIGNS:
                continue
            if not campaign.get('available'):
                audit['pending_inventory'].append({'campaign': key, 'reason': 'Campaign data unavailable'})
                continue
            check_revocation(state, engine)
            destination = output / (key + '.json')
            previous = json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else {'rows': []}
            baseline = {(r['run_id'], r['proposal'], r['candidate_id']): r for r in previous['rows']}
            records, jobs = [], {}
            retained_index = None
            processor = processor_version(key)
            revoked = processor_revocation(state, processor)
            if revoked:
                audit['pending_inventory'].append({'campaign': key, 'processor_sha256': processor, 'reason': revoked})
                save_audit(audit, state, output)
                continue
            for run in campaign['runs']:
                root = Path(campaign['campaign']) / 'runs' / run['run_id']
                if root.resolve().parent != (Path(campaign['campaign']) / 'runs').resolve():
                    raise ValueError('Run identity escapes campaign')
                paths = {}
                def candidate(relative):
                    if relative not in paths:
                        paths[relative] = safe_candidate(root, relative)
                    return paths[relative]
                for point in run['points']:
                    if not point.get('candidate_id'):
                        audit['pending_inventory'].append({'campaign': key, 'run_id': run['run_id'],
                                                           'proposal': point.get('proposal'),
                                                           'reason': 'Proposal has no candidate source identity yet'})
                        continue
                    source_error = None
                    candidate_path = None
                    try:
                        relative = point.get('artifact_path') or 'candidates/' + point['candidate_id']
                        candidate_path = candidate(relative)
                        # A campaign event may retain a stale artifact_path after a
                        # failed materialization.  Never review a different
                        # candidate's Python as though it belonged to this ID.
                        if Path(relative).name != point['candidate_id']:
                            raise ValueError('Artifact path candidate identity mismatch: expected ' +
                                             point['candidate_id'] + ', found ' + Path(relative).name)
                        source = cache.read(candidate_path, capture=True)
                        parents = []
                    except (OSError, ValueError, UnicodeError) as exc:
                        source, parents, source_error = {}, [], str(exc)
                    identity = (run['run_id'], point['proposal'], point['candidate_id'])
                    row = dict(baseline.get(identity) or blank_row(run, point, source, engine))
                    if row.get('source_sha256') != sha(source):
                        row = blank_row(run, point, source, engine)
                    if (row.get('classification') in {'changing', 'mixed'} and
                            row.get('reviewer') != 'Task-specific semantic review'):
                        row.update(classification='uncertain', family=None,
                                   notes='Earlier component-presence comparison requires a source-traced mechanism witness.')
                    queue_hashes = queue_parent_hashes(snapshot, key, run, point, source) if source else None
                    parent_unavailable_provenance = None
                    if queue_hashes is not None:
                        if retained_index is None:
                            retained_index = retained_sources_by_sha(campaign['campaign'], cache)
                        resolved = [retained_index.get(parent_hash, []) for parent_hash in queue_hashes]
                        parents = [sources[0][1] if sources else {} for sources in resolved]
                        parent_hashes = list(queue_hashes)
                        parent_unavailable_provenance = {
                            'kind': 'queue_recorded_parent_source_unavailable',
                            'candidate_id': point['candidate_id'],
                            'run_id': run['run_id'],
                            'proposal': point['proposal'],
                            'source_sha256': sha(source),
                            'parent_source_sha256': [parent_hash for parent_hash, sources in zip(queue_hashes, resolved)
                                                      if not sources],
                        }
                    else:
                        try:
                            parents = [cache.read(candidate('candidates/' + parent), capture=True)
                                       for parent in point.get('parent_ids', [])]
                        except (OSError, ValueError, UnicodeError) as exc:
                            parents, source_error = ([{}] * len(point.get('parent_ids', []))), str(exc)
                        parent_hashes = [sha(p) for p in parents]
                    if 'parent_source_sha256' in row and row['parent_source_sha256'] != parent_hashes:
                        row.update(classification='uncertain', notes='Recorded parent source changed; transition requires a fresh comparison.')
                    row['parent_source_sha256'] = parent_hashes
                    row['source_path'] = str(candidate_path) if candidate_path else None
                    row['review_pipeline_sha256'] = engine
                    row.pop('semantic_review_error', None)
                    row.pop('classification_before_consistency_check', None)
                    syntax_error = None
                    if source:
                        try:
                            for text in source.values():
                                ast.parse(text)
                        except SyntaxError as exc:
                            syntax_error = str(exc)
                    invalid = syntax_error and point.get('failure_kind') == 'source_preflight' and not point.get('valid')
                    job_keys = []
                    if invalid:
                        row.update(classification='invalid_source', invalid_kind='syntax_error', fingerprint={}, fingerprint_complete=False, family=None,
                                   implemented=False, executable=False, notes='Invalid Python source independently confirmed by campaign preflight: ' + syntax_error)
                    elif source and not syntax_error and any(not parent for parent in parents):
                        missing = [parent_id for parent_id, parent in zip(point.get('parent_ids', []), parents)
                                   if not parent]
                        row.update(classification='parent_source_unavailable', fingerprint={}, training={}, inference={}, settings={},
                                   fingerprint_complete=False, family=None,
                                   implemented=None, executable=None,
                                   notes='Queue-recorded parent source unavailable; no parent-child ontology transition can be claimed.'
                                   if parent_unavailable_provenance else
                                   'Recorded parent source unavailable; no parent-child ontology transition can be claimed.',
                                   parent_source_unavailable_provenance=parent_unavailable_provenance or {
                                       'kind': 'recorded_parent_source_unavailable',
                                       'candidate_id': point['candidate_id'],
                                       'parent_ids': missing,
                                       'parent_artifact_paths': ['candidates/' + parent_id for parent_id in missing],
                                   })
                        row.pop('family_signature', None)
                        row.pop('fingerprint_provenance', None)
                    elif source and not syntax_error:
                        for parent in parents or ([source] if point.get('is_seed') else []):
                            if not parent:
                                continue
                            pair = sha([key, sha(parent), sha(source), processor])
                            jobs.setdefault(pair, (key, parent, source,
                                                   str(state / 'processor-cache' / processor / (pair + '.json')), engine, processor))
                            job_keys.append(pair)
                        if point.get('is_seed') or len(parents) == 1 and source == parents[0]:
                            row.update(classification='preserving', notes='Seed reference; no transition.' if point.get('is_seed') else 'Identical Python source to the recorded parent; no implemented source edit.')
                        row['implemented'] = bool(point.get('is_seed') or not parents or source != parents[0])
                        row['executable'] = True if point.get('valid') or point.get('failure_kind') == 'nonqualification' else row.get('executable')
                    elif source:
                        row.update(classification='uncertain', fingerprint_complete=False, family=None,
                                   notes=syntax_error or 'Candidate source needs semantic review.')
                    else:
                        unavailable = source_error or 'Candidate source contains no readable Python files.'
                        row.update(classification='source_unavailable', fingerprint={}, training={}, inference={}, settings={},
                                   fingerprint_complete=False, family=None,
                                   implemented=None, executable=None,
                                   notes='Source unavailable: ' + unavailable,
                                   source_unavailable_provenance={
                                       'kind': 'candidate_source_unavailable',
                                       'candidate_id': point['candidate_id'],
                                       'artifact_path': point.get('artifact_path') or 'candidates/' + point['candidate_id'],
                                       'error': unavailable,
                                   })
                        row.pop('family_signature', None)
                        row.pop('fingerprint_provenance', None)
                    records.append((run, point, row, job_keys, source, parents))
                    if len(records) % 250 == 0:
                        print(key, 'indexed', len(records), 'records', flush=True)
            print(key, 'source pairs', len(jobs), 'records', len(records), flush=True)
            cache.db.commit()
            with ProcessPoolExecutor(max_workers=workers) as pool:
                answers = {}
                for index, (pair, answer) in enumerate(zip(jobs, pool.map(review_job, jobs.values(), chunksize=12)), 1):
                    answers[pair] = answer
                    if index % 250 == 0:
                        print(key, 'reviewed', index, '/', len(jobs), 'source pairs', flush=True)
            for run, point, row, job_keys, source, parents in records:
                outcomes = [answers[j]['review'] for j in job_keys]
                usable = [r for r in outcomes if r and not r.get('error')]
                row['parent_reviews'] = [
                    {'source_sha256': sha(jobs[pair][1]), 'classification': answer['classification'],
                     'changed_components': answer.get('changed_components', []), 'evidence': answer.get('evidence', [])}
                    for pair, answer in zip(job_keys, outcomes) if answer and not answer.get('error')]
                if usable and len(usable) == len(outcomes):
                    result = dict(usable[0])
                    if any(r['classification'] in {'changing', 'mixed'} for r in usable):
                        result = dict(next(r for r in usable if r['classification'] in {'changing', 'mixed'}))
                    elif any(r['classification'] == 'uncertain' for r in usable):
                        result = dict(next(r for r in usable if r['classification'] == 'uncertain'))
                    if any(r.get('fingerprint') != usable[0].get('fingerprint') for r in usable):
                        result = None
                        row.update(classification='uncertain', fingerprint_complete=False, family=None,
                                   notes='Candidate fingerprint requires review: parent comparisons describe the same source differently.',
                                   semantic_review_error='Parent comparisons disagree about the same candidate fingerprint')
                    if result:
                        apply_result(row, result, key, engine, point)
                elif any(r and r.get('error') for r in outcomes):
                    row['semantic_review_error'] = [r['error'] for r in outcomes if r and r.get('error')]
                required = CORE + TASK_KEYS[task_for(key)].split()
                row['fingerprint'] = {k: row.get('fingerprint', {}).get(k) for k in required} | row.get('fingerprint', {})
            canonicalize_exact_source_fingerprints([record[2] for record in records])
            anchors, conflicts = {}, set()
            for _, _, row, _, _, _ in records:
                if row.get('fingerprint_complete'):
                    source_hash = row['source_sha256']
                    if source_hash in anchors and anchors[source_hash]['fingerprint'] != row['fingerprint']:
                        conflicts.add(source_hash)
                    anchors[source_hash] = row
            for _, _, row, _, _, _ in records:
                if row['source_sha256'] in conflicts:
                    row.update(classification='uncertain', fingerprint_complete=False, family=None,
                               notes='Candidate fingerprint requires review: identical source received different complete fingerprints.',
                               semantic_review_error='Conflicting complete fingerprints for identical source')
            family_conflicts = reconcile_families([record[2] for record in records])
            queue = []
            for run, point, row, _, _, _ in records:
                anchor = anchors.get(row['source_sha256'])
                if anchor and row['source_sha256'] not in conflicts and not row.get('fingerprint_complete') and row['classification'] != 'invalid_source':
                    row.update(fingerprint=dict(anchor['fingerprint']), fingerprint_complete=True, family=anchor.get('family'),
                               fingerprint_provenance={'kind': 'exact reviewed source match', 'source_sha256': row['source_sha256'],
                                                       'anchor_candidate_id': anchor['candidate_id']})
                if (row['classification'] not in RESOLVED or
                        not row.get('fingerprint_complete') and row['classification'] not in {
                            'invalid_source', 'source_unavailable', 'parent_source_unavailable'}):
                    queue.append(dict(run_id=run['run_id'], proposal=point['proposal'], candidate_id=point['candidate_id'],
                                      source_sha256=row['source_sha256'], parent_source_sha256=row['parent_source_sha256'],
                                      classification=row['classification'], fingerprint_complete=bool(row.get('fingerprint_complete')),
                                      notes=row['notes'], error=row.get('semantic_review_error'),
                                      source_path=row['source_path']))
            doc = dict(schema_version='1.0', campaign=campaign['campaign'], task=task_for(key), generated_at=utcnow(),
                       rubric='Task-specific semantic ontology review · source-bound rules and direct references',
                       review_snapshot=snapshot.get('generated_at'), review_pipeline_sha256=engine,
                       rows=[r[2] for r in records])
            summary, _ = validate(doc, campaign, task_for(key))
            summary['processor_sha256'] = processor
            summary['family_consistency_conflicts'] = len(family_conflicts)
            summary['fingerprint_consistency_conflicts'] = len(conflicts)
            atomic_json(state / 'queue' / (key + '.json'), queue)
            revoked = processor_revocation(state, processor)
            if revoked:
                audit['pending_inventory'].append({'campaign': key, 'processor_sha256': processor, 'reason': revoked})
                print(key, 'publication withheld:', revoked, flush=True)
            else:
                publish_campaign(key, doc, state, output, engine)
                audit['campaigns'][key] = summary
                audit['pending_campaigns'].remove(key)
            save_audit(audit, state, output)
            print(key, summary, flush=True)
            cache.captured.clear()
    finally:
        cache.close()
        save_audit(audit, state, output)
    return audit


def freeze_engine(state):
    directory = Path(__file__).resolve().parent
    files = {p.name: p.read_bytes() for p in directory.glob('*.py')
             if p.name.startswith(('ontology_', 'review_ontology_', 'adjudicate_ontology_', 'publish_ontology_', 'apply_ontology_'))}
    files.update({p.name: p.read_bytes() for p in directory.glob('ontology_*.json')})
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    version = sha(hashes)[:16]
    root = state / 'engines' / version
    if not (root / 'manifest.json').exists():
        (root / 'experiments').mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (root / 'experiments' / name).write_bytes(content)
        (root / 'experiments' / '__init__.py').write_text('', encoding='utf-8')
        atomic_json(root / 'manifest.json', {'version': version, 'files': hashes})
    return root, version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, default=Path('outputs/ontology/live'))
    parser.add_argument('--output', type=Path, default=Path('outputs/ontology'))
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:8765/api/data')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--engine', default='')
    args = parser.parse_args()
    state, output = args.state.resolve(), args.output.resolve()
    state.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        process_snapshot(args.snapshot, output, state, args.engine, args.workers)
        return
    while True:
        started = time.monotonic()
        lock = None
        try:
            lock = acquire_lock(state)
            if args.snapshot:
                snapshot = json.loads(args.snapshot.read_text(encoding='utf-8'))
            else:
                with urllib.request.urlopen(args.url, timeout=90) as response:
                    snapshot = json.load(response)
            # The live queue is durable provenance for historical reviews.  Put
            # it in the snapshot before freezing the worker so parent selection
            # cannot depend on mutable state after this point.
            snapshot = queue_provenance_for_snapshot(snapshot, state)
            atomic_json(state / 'snapshot.json', snapshot)
            engine_root, version = freeze_engine(state)
            atomic_json(state / 'status.json', {'pid': os.getpid(), 'phase': 'reviewing', 'started_at': utcnow(), 'engine': version})
            command = [sys.executable, '-u', '-m', 'experiments.ontology_review_service', '--worker',
                       '--snapshot', str(state / 'snapshot.json'), '--state', str(state), '--output', str(output),
                       '--workers', str(args.workers), '--engine', version]
            result = subprocess.run(command, cwd=engine_root, check=False)
            if result.returncode:
                raise RuntimeError('Review worker failed with exit code ' + str(result.returncode))
            audit = json.loads((state / 'audit.json').read_text(encoding='utf-8'))
            atomic_json(state / 'status.json', {'pid': os.getpid(), 'phase': 'waiting' if args.watch else 'completed',
                                             'checked_at': utcnow(), 'engine': version,
                                             'transitions_complete': audit['transitions_complete'],
                                             'fingerprints_complete': audit['fingerprints_complete']})
        except ReviewBusy as exc:
            print(str(exc), flush=True)
        except Exception as exc:
            atomic_json(state / 'status.json', {'pid': os.getpid(), 'phase': 'failed', 'checked_at': utcnow(), 'error': str(exc)})
            if not args.watch:
                raise
            print('Review retry pending:', str(exc), flush=True)
        finally:
            if lock:
                lock.close()
        if not args.watch:
            break
        time.sleep(max(1, args.interval - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
