"""Apply directly inspected full fingerprints to exact source hashes."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.review_ontology_sources import read_sources
from experiments.review_ontology_v2 import atomic_json


def apply(snapshot_path, reviews, references):
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    reference_doc = json.loads(references.read_text(encoding='utf-8'))
    reference_hash = hashlib.sha256(references.read_bytes()).hexdigest()
    archive = reviews.parent / 'reference-versions' / (reference_hash + '.json')
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(references.read_bytes())
    results = {}
    for key in {r['campaign_key'] for r in reference_doc['references']}:
        campaign = snapshot['campaigns'][key]
        doc = json.loads((reviews / (key + '.json')).read_text(encoding='utf-8'))
        rowmap = {(r['run_id'], r['proposal']): r for r in doc['rows']}
        pointmap = {(r['run_id'], p['proposal']): p for r in campaign['runs'] for p in r['points']}
        refs = [r for r in reference_doc['references'] if r['campaign_key'] == key]
        exact = {}
        for ref in refs:
            row = rowmap[(ref['run_id'], ref['proposal'])]
            point = pointmap[(ref['run_id'], ref['proposal'])]
            if row['candidate_id'] != ref['candidate_id'] or row['source_sha256'] != ref['source_sha256']:
                raise ValueError('Reference does not match the reviewed candidate')
            root = Path(campaign['campaign']) / 'runs' / ref['run_id']
            source = read_sources(root / (point.get('artifact_path') or 'candidates/' + point['candidate_id']))
            parent = read_sources(root / 'candidates' / point['parent_ids'][0])
            digest = lambda data: hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            if digest(source) != ref['source_sha256'] or digest(parent) != ref['parent_source_sha256']:
                raise ValueError('Reference source or recorded parent changed')
            seed = next(r for r in doc['rows'] if r['source_sha256'] == ref['family_reference_seed_sha256'])
            exact[ref['source_sha256']] = (ref, seed['family'])
            row.update(classification=ref['classification'], notes=ref['rationale'],
                       reviewer=reference_doc['reviewer'], changed_components=[])
        direct = 0
        for row in doc['rows']:
            matched = exact.get(row['source_sha256'])
            if not matched:
                continue
            ref, family = matched
            row.update(fingerprint=dict(ref['fingerprint']), fingerprint_complete=True, family=family,
                       fingerprint_provenance={'kind': 'direct source-specific semantic reference',
                                               'source_sha256': ref['source_sha256'],
                                               'reference_file_sha256': reference_hash,
                                               'rationale': ref['rationale']})
            direct += 1
        # Only settings matches propagate full component details. Generic
        # preserving labels alone never license this operation.
        anchors, conflicts = {}, set()
        for row in doc['rows']:
            signature = row.get('settings_shape_sha256')
            if not row.get('fingerprint_complete') or not signature:
                continue
            if signature in anchors and anchors[signature]['fingerprint'] != row['fingerprint']:
                conflicts.add(signature)
            anchors[signature] = row
        for signature in conflicts:
            anchors.pop(signature, None)
        propagated = 0
        for row in doc['rows']:
            anchor = anchors.get(row.get('settings_shape_sha256'))
            if anchor and not row.get('fingerprint_complete') and row['classification'] != 'invalid_source':
                row.update(fingerprint=dict(anchor['fingerprint']), fingerprint_complete=True,
                           family=anchor['family'], fingerprint_provenance={
                               'kind': 'settings-normalized match to complete reference',
                               'source_sha256': anchor['source_sha256'],
                               'reference_file_sha256': reference_hash})
                propagated += 1
        doc['semantic_references'] = {'sha256': reference_hash, 'count': len(refs),
                                      'direct_fingerprints': direct, 'newly_propagated_fingerprints': propagated,
                                      'propagated_fingerprints': sum(r.get('fingerprint_provenance', {}).get('kind') == 'settings-normalized match to complete reference' for r in doc['rows']),
                                      'signature_conflicts_not_propagated': len(conflicts)}
        if 'source-specific semantic references' not in doc['rubric']:
            doc['rubric'] += ' · source-specific semantic references'
        atomic_json(reviews / (key + '.json'), doc)
        results[key] = doc['semantic_references']
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, default=Path('outputs/ontology/v2/dashboard-snapshot-latest.json'))
    parser.add_argument('--reviews', type=Path, default=Path('outputs/ontology/v2/final'))
    parser.add_argument('--references', type=Path, default=Path('experiments/ontology_reference_reviews.json'))
    args = parser.parse_args()
    print(json.dumps(apply(args.snapshot, args.reviews, args.references), indent=2))
