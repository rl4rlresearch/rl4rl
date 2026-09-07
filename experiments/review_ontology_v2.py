"""Resumable corpus audit, with cached source proofs and explicit residual cases."""
import argparse
import ast
import collections
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from experiments.ontology_semantics import fingerprint_program, program
from experiments.review_ontology_sources import read_sources, task_for, analyze, CORE, TASK_KEYS


def atomic_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    tmp.replace(path)


def worker(job):
    sha,sources,task,cache=job
    file=Path(cache)/(task+'-'+sha+'.json')
    if file.exists():
        return sha,json.loads(file.read_text(encoding='utf-8'))
    try:
        proof=fingerprint_program(sources)
        # Run category extraction on the selected configuration, not all declared branches.
        definitions,locations,*_=program(sources)
        specialized={}
        for filename,text in sources.items():
            tree=ast.parse(text)
            tree.body=[definitions[n.name] if isinstance(n,(ast.ClassDef,ast.FunctionDef)) else n for n in tree.body]
            specialized[filename]=ast.unparse(ast.fix_missing_locations(tree))
        profile=analyze(specialized,task)
        # Keep original line numbers through AST nodes for the execution trace.
        profile['evidence_note']='Category snippets refer to configuration-specialized source; trace ranges refer to original source.'
        if profile.get('error'):
            proof['category_error']=profile['error']
        result={'proof':proof,'profile':profile,'error':None}
    except (SyntaxError,ValueError,RecursionError,TypeError,AttributeError) as exc:
        result={'proof':{},'profile':{},'error':type(exc).__name__+': '+str(exc)}
    atomic_json(file,result)
    return sha,result


def run(snapshot_path,output,campaign_keys=None):
    snapshot=json.loads(snapshot_path.read_text(encoding='utf-8'))
    code_hash=hashlib.sha256(b''.join(Path('experiments',name).read_bytes() for name in ['ontology_semantics.py','review_ontology_sources.py','review_ontology_v2.py'])).hexdigest()[:12]
    cache=output/'cache'/code_hash
    cache.mkdir(parents=True,exist_ok=True)
    summary={}
    for key,campaign in snapshot['campaigns'].items():
        if not campaign.get('available') or campaign_keys and key not in campaign_keys:continue
        paths={};points=[]
        for r in campaign['runs']:
            root=Path(campaign['campaign'])/'runs'/r['run_id']
            for p in r['points']:
                if not p.get('candidate_id'):continue
                path=str(root/(p.get('artifact_path') or 'candidates/'+p['candidate_id']).replace('\\','/'))
                parents=[str(root/'candidates'/cid) for cid in p.get('parent_ids',[])]
                for item in [path]+parents:paths[item]=Path(item)
                points.append((r,p,path,parents))
        bundles={};hashes={}
        print(key,'read',len(paths),'sources',flush=True)
        with ThreadPoolExecutor(max_workers=12) as pool:
            for path,source in zip(paths,pool.map(read_sources,paths.values())):
                sha=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
                hashes[path]=sha if source else None
                if source:bundles[sha]=source
        profiles={}
        with ProcessPoolExecutor(max_workers=4) as pool:
            jobs=((sha,source,task_for(key),str(cache)) for sha,source in bundles.items())
            for i,(sha,profile) in enumerate(pool.map(worker,jobs,chunksize=8)):
                profiles[sha]=profile
                if i and i%250==0:print(key,'traced',i,'/',len(bundles),flush=True)
        rows=[];residuals=[]
        for r,p,path,parents in points:
            sha=hashes[path]
            result=profiles.get(sha,{'error':'Source artifacts unavailable','proof':{},'profile':{}})
            profile,proof=result['profile'],result['proof']
            parent=profiles.get(hashes.get(parents[0])) if len(parents)==1 else None
            classification='uncertain';reason='Residual source difference needs semantic adjudication.'
            changed=[]
            if result['error']:
                classification='unannotated';reason=result['error']
                if result['error'].startswith('SyntaxError:') and p.get('failure_kind')=='source_preflight' and not p.get('valid'):
                    classification='invalid_source'
                    reason='Source is syntactically invalid, independently confirmed by campaign source preflight. No executable architecture exists for this proposal.'
            elif p.get('is_seed'):
                classification='preserving';reason='Seed reference; no transition.'
            elif parent and sha==hashes.get(parents[0]):
                classification='preserving';reason='Identical source to recorded parent; no implemented source edit.'
            elif parent and proof.get('shape')==parent['proof'].get('shape') and proof.get('shape'):
                classification='preserving';reason='Configuration-resolved computation unchanged under documented size/bias/coordinate normalization.'
            elif parent and not profile.get('error') and not parent['profile'].get('error'):
                before=parent['profile'].get('family_signature',{});after=profile.get('family_signature',{})
                changed=sorted(k for k in before.keys()&after.keys() if before[k]!=after[k])
                if changed:
                    classification='changing';reason='Resolved mechanism difference: '+', '.join(changed)+'.'
            row={'run_id':r['run_id'],'proposal':p['proposal'],'candidate_id':p['candidate_id'],
                 'family':profile.get('family') if classification in {'preserving','changing'} else None,
                 'classification':classification,'proposed_change':None,
                 'implemented':False if classification=='invalid_source' else None if result['error'] else p.get('is_seed',False) or not parent or sha!=hashes.get(parents[0]),
                 'executable':False if classification=='invalid_source' else True if p.get('valid') or p.get('failure_kind')=='nonqualification' else None,
                 'fingerprint':profile.get('fingerprint',dict.fromkeys(CORE+TASK_KEYS[task_for(key)].split())),
                 'training':profile.get('training',{}),'inference':profile.get('inference',{}),'settings':profile.get('settings',{}),
                 'reviewer':'Codex source-program audit v2','notes':reason,'source_sha256':sha,
                 'component_evidence':profile.get('evidence',{}),'changed_components':changed,
                 'review_method':'Configuration specialization and rooted computational comparison',
                 'execution_shape_sha256':proof.get('shape'),'source_trace':proof.get('coverage',[]),
                 'config_decisions':proof.get('config_decisions',[]),'resolved_config':proof.get('resolved_config',{}),
                 'review_cache':str(cache/(task_for(key)+'-'+str(sha)+'.json'))}
            rows.append(row)
            if classification in {'uncertain','unannotated'}:
                residuals.append({'run_id':r['run_id'],'proposal':p['proposal'],'path':path,'parents':parents,
                                  'source_sha256':sha,'reason':reason,'proof_shape':proof.get('shape')})
        doc={'schema_version':'1.0','campaign':campaign['campaign'],'rubric':'Source-program audit v2 · configuration-resolved; residuals explicitly audited',
             'generated_at':datetime.now(timezone.utc).isoformat(),'review_code_sha256':code_hash,'task':task_for(key),'rows':rows}
        atomic_json(output/(key+'.json'),doc)
        atomic_json(output/(key+'-residuals.json'),residuals)
        summary[key]={'rows':len(rows),'classifications':dict(collections.Counter(r['classification'] for r in rows)),
                      'unique_residual_sources':len({r['source_sha256'] for r in residuals}),
                      'unique_residual_shapes':len({r['proof_shape'] for r in residuals})}
        atomic_json(output/'summary.json',summary)
        print(key,summary[key],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,default=Path('outputs/ontology/dashboard-snapshot.json'))
    parser.add_argument('--output',type=Path,default=Path('outputs/ontology/v2'))
    parser.add_argument('--campaign',action='append')
    args=parser.parse_args()
    run(args.snapshot,args.output,args.campaign)
