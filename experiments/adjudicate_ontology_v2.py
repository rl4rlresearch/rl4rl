"""Apply independent settings proofs and hash-locked seed reference reviews."""
import argparse
import collections
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from experiments.ontology_semantics import fingerprint_program
from experiments.ontology_transition_review import settings_signature
from experiments.ontology_affine_readout import affine_readout_signature
from experiments.ontology_seed_rubric import seed_fingerprint, AUDITED_SEED_HASHES
from experiments.review_ontology_sources import read_sources
from experiments.review_ontology_v2 import atomic_json


def proof_job(job):
    sha,source,cache=job
    file=Path(cache)/(sha+'.json')
    if file.exists():return sha,json.loads(file.read_text(encoding='utf-8'))
    try:
        result={'shape':fingerprint_program(source)['shape'],'settings_shape':settings_signature(source),
                'affine_readout_shape':affine_readout_signature(source)}
    except (SyntaxError,ValueError,RecursionError,TypeError,AttributeError) as e:
        result={'error':type(e).__name__+': '+str(e)}
    atomic_json(file,result)
    return sha,result


def adjudicate(snapshot_path,input_dir,campaign_filter=None):
    snapshot=json.loads(snapshot_path.read_text(encoding='utf-8'))
    output=input_dir/'final'
    proofhash=hashlib.sha256(b''.join(Path('experiments',n).read_bytes() for n in ['ontology_semantics.py','ontology_transition_review.py','ontology_affine_readout.py'])).hexdigest()[:12]
    codehash=hashlib.sha256(b''.join(Path('experiments',n).read_bytes() for n in ['ontology_semantics.py','ontology_transition_review.py','ontology_affine_readout.py','ontology_seed_rubric.py','adjudicate_ontology_v2.py'])).hexdigest()[:12]
    cache=input_dir/'proof-cache'/proofhash
    summary={}
    for key,campaign in snapshot['campaigns'].items():
        if not campaign.get('available') or campaign_filter and key not in campaign_filter:continue
        doc=json.loads((input_dir/(key+'.json')).read_text(encoding='utf-8'))
        points={(r['run_id'],p['proposal']):(r,p) for r in campaign['runs'] for p in r['points']}
        rowmap={(r['run_id'],r['proposal']):r for r in doc['rows']}
        paths={};rowpaths={};source_hashes={};bundles={}
        for row in doc['rows']:
            run,point=points[(row['run_id'],row['proposal'])]
            root=Path(campaign['campaign'])/'runs'/run['run_id']
            path=str(root/(point.get('artifact_path') or 'candidates/'+row['candidate_id']).replace('\\','/'))
            parents=[str(root/'candidates'/cid) for cid in point.get('parent_ids') or []]
            rowpaths[(row['run_id'],row['proposal'])]=(path,parents)
            for p in [path]+parents:paths[p]=Path(p)
        print(key,'read proofs',len(paths),flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for path,sources in zip(paths,pool.map(read_sources,paths.values())):
                sha=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
                source_hashes[path]=sha if sources else None
                if sources:bundles[sha]=sources
        proofs={}
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i,(sha,proof) in enumerate(pool.map(proof_job,((sha,source,str(cache)) for sha,source in bundles.items()),chunksize=12)):
                proofs[sha]=proof
                if i and i%500==0:print(key,'proofs',i,'/',len(bundles),flush=True)
        promoted=0
        for row in doc['rows']:
            path,parents=rowpaths[(row['run_id'],row['proposal'])]
            proof=proofs.get(source_hashes[path],{})
            if source_hashes[path]!=row['source_sha256']:
                row.update(classification='unannotated',family=None,notes='Source changed or disappeared after extraction; previous label withheld.')
                continue
            row['settings_shape_sha256']=proof.get('settings_shape')
            if row['classification']=='unannotated' and row['notes'].startswith(('SyntaxError:','IndentationError:','TabError:')):
                _,point=points[(row['run_id'],row['proposal'])]
                if point.get('failure_kind')=='source_preflight' and not point.get('valid'):
                    row.update(classification='invalid_source',implemented=False,executable=False,
                               notes='Syntactically invalid source, independently confirmed by the campaign preflight; no executable architecture for this proposal.')
            if len(parents)==1 and row['classification'] not in {'invalid_source','unannotated'}:
                before=proofs.get(source_hashes[parents[0]],{})
                exact=proof.get('shape') and proof['shape']==before.get('shape')
                numeric=proof.get('settings_shape') and proof['settings_shape']==before.get('settings_shape')
                affine=proof.get('affine_readout_shape') and proof['affine_readout_shape']==before.get('affine_readout_shape')
                identical=row['source_sha256']==source_hashes[parents[0]]
                if exact or numeric or affine or identical:
                    if row['classification']!='preserving':promoted+=1
                    row['classification']='preserving'
                    row['changed_components']=[]
                    row['notes']='Preserving proof: '+('identical source.' if identical else 'same configuration-resolved normalized computation.' if exact else 'same operations, branches and coordinate selections; only nonzero numeric settings or signal-independent sampling changed.' if numeric else 'same recurrent/core computation and same readout features; only independent affine head coordinates and numerical settings changed.')
                elif row['classification']=='preserving':
                    row['classification']='uncertain';row['family']=None
                    row['notes']='Earlier match did not survive the full configuration/factory guard; residual difference retained for semantic review.'
            row['adjudication_code_sha256']=codehash
        # Apply a reference fingerprint only to a byte-identical reviewed seed.
        seedrows=[row for row in doc['rows'] if points[(row['run_id'],row['proposal'])][1].get('is_seed')]
        reference=next((r for r in seedrows if r['source_sha256']==AUDITED_SEED_HASHES.get(key)),None)
        known={}
        if reference:
            for row in seedrows:
                if row['source_sha256']==reference['source_sha256']:
                    row['fingerprint']=seed_fingerprint(key)
                    row['fingerprint_complete']=True
                    row['fingerprint_provenance']={'kind':'hash-locked audited seed','source_sha256':reference['source_sha256']}
        by_candidate=collections.defaultdict(list)
        for row in sorted(doc['rows'],key=lambda r:r['proposal']):by_candidate[(row['run_id'],row['candidate_id'])].append(row)
        # Forward inheritance through recorded parents, never chronological neighbors.
        for _ in range(3):
            for row in sorted(doc['rows'],key=lambda r:r['proposal']):
                _,point=points[(row['run_id'],row['proposal'])]
                parents=point.get('parent_ids') or []
                history=by_candidate.get((row['run_id'],parents[0]),[]) if len(parents)==1 else []
                parent=next((r for r in reversed(history) if r['proposal']<row['proposal']),None)
                if row['classification']=='preserving' and parent and parent.get('family'):
                    row['family']=parent['family']
                    # A preserving coordinate rewrite can still change sharing or
                    # parameter-construction details. Do not copy a full fingerprint
                    # unless only numeric settings/sampling (or no source) changed.
                    same_details=row['source_sha256']==parent['source_sha256'] or (row.get('settings_shape_sha256') and row['settings_shape_sha256']==parent.get('settings_shape_sha256'))
                    if parent.get('fingerprint_complete') and same_details:
                        row['fingerprint']=dict(parent['fingerprint']);row['fingerprint_complete']=True
                        row['fingerprint_provenance']={'kind':'settings-only transition proof','parent_id':parent['candidate_id'],'parent_proposal':parent['proposal']}
                signature=row.get('settings_shape_sha256')
                if row.get('fingerprint_complete') and signature:known.setdefault(signature,row)
                elif signature in known:
                    anchor=known[signature]
                    row['fingerprint']=dict(anchor['fingerprint']);row['family']=anchor['family'];row['fingerprint_complete']=True
                    row['fingerprint_provenance']={'kind':'matching settings-normalized program','candidate_id':anchor['candidate_id'],'run_id':anchor['run_id']}
        doc['adjudication']={'code_sha256':codehash,'promoted_preserving':promoted,'reference_seed_sha256':reference and reference['source_sha256']}
        doc['rubric']='Source-program audit v2 · branch resolution, settings proofs and hash-locked seed references; remaining residuals explicit'
        atomic_json(output/(key+'.json'),doc)
        summary[key]={'rows':len(doc['rows']),'classifications':dict(collections.Counter(r['classification'] for r in doc['rows'])),
                      'complete_fingerprints':sum(bool(r.get('fingerprint_complete')) for r in doc['rows']),'new_preserving_proofs':promoted}
        atomic_json(output/(key+'-summary.json'),summary[key])
        print(key,summary[key],flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,default=Path('outputs/ontology/v2/dashboard-snapshot.json'))
    parser.add_argument('--input',type=Path,default=Path('outputs/ontology/v2'))
    parser.add_argument('--campaign',action='append')
    args=parser.parse_args()
    adjudicate(args.snapshot,args.input,args.campaign)
