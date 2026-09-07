"""Run the recurrent task review without modifying published dashboard data."""
import argparse
import collections
import json
import hashlib
from pathlib import Path
from experiments.ontology_review_recurrent import review_pair, profile, _cached_profile, _json

def read_sources(path):
    return {f.relative_to(path).as_posix():f.read_text(encoding='utf-8-sig') for f in sorted(path.rglob('*.py')) if f.is_file()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign');parser.add_argument('--limit',type=int);parser.add_argument('--suffix',default='current')
    parser.add_argument('--input-residuals',type=Path);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    out=Path('outputs/ontology/semantic/recurrent');out.mkdir(parents=True,exist_ok=True)
    snapshot=json.loads(Path('outputs/ontology/live/snapshot.json').read_text(encoding='utf-8'))
    engine_files=[Path('experiments')/n for n in ['ontology_review_recurrent.py','ontology_recurrent_readout.py','ontology_recurrent_equivalence.py','ontology_recurrent_branches.py','ontology_recurrent_references.json']]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in engine_files}
    selected=None
    if args.input_residuals:
        selected={(r['run_id'],r['proposal']) for r in json.loads(args.input_residuals.read_text(encoding='utf-8'))}
    for key in ['uci_har_pareto_v21','openevolve_v21_tiny_kws_rnn']:
        if args.campaign and key!=args.campaign:continue
        c=snapshot['campaigns'][key];old=json.loads(Path('outputs/ontology/'+key+'.json').read_text())['rows']
        known={(r['run_id'],r['proposal']) for r in old}
        old += [dict(run_id=r['run_id'],proposal=p['proposal'],classification='uncertain') for r in c['runs'] for p in r['points'] if (r['run_id'],p['proposal']) not in known]
        points={(r['run_id'],p['proposal']):(r,p) for r in c['runs'] for p in r['points']}
        rows=[];residual=[];counts=collections.Counter();errors=collections.Counter();sources={};completed=set()
        checkpoint=out/(key+'-'+args.suffix+'-checkpoint.jsonl');manifest=out/(key+'-'+args.suffix+'-manifest.json')
        if args.resume and checkpoint.exists():
            if json.loads(manifest.read_text(encoding='utf-8'))['source_rule_hashes']!=hashes:
                raise ValueError('Reviewer files changed; choose a new suffix for the next audit version')
            for line in checkpoint.read_text(encoding='utf-8').splitlines():
                saved=json.loads(line);record=saved['record'];completed.add((record['run_id'],record['proposal']))
                if saved['kind']=='decision':
                    rows.append(record);counts[record['classification']]+=1;counts['fingerprint_complete']+=record['fingerprint_complete']
                else:
                    residual.append(record);errors.update(record.get('errors',{}).values())
                    if record.get('error'):errors[record['error']]+=1
        else:
            checkpoint.write_text('',encoding='utf-8');manifest.write_text(json.dumps({'source_rule_hashes':hashes},indent=2),encoding='utf-8')
        def save(kind,record):
            with checkpoint.open('a',encoding='utf-8') as handle:handle.write(json.dumps({'kind':kind,'record':record})+'\n')
        for row in old:
            if row['classification']!='uncertain':continue
            identity=(row['run_id'],row['proposal'])
            if identity in completed or selected is not None and identity not in selected:continue
            if args.limit and len(rows)+len(residual)>=args.limit:break
            r,p=points[(row['run_id'],row['proposal'])]
            if len(p.get('parent_ids') or [])!=1:continue
            root=Path(c['campaign'])/'runs'/r['run_id']
            parent=root/'candidates'/p['parent_ids'][0];child=root/(p.get('artifact_path') or 'candidates/'+p['candidate_id'])
            def source(path):
                k=str(path)
                if k not in sources:sources[k]=read_sources(path)
                return sources[k]
            try:
                a,b=source(parent),source(child);result=review_pair(a,b,key)
                if result and result['classification']!='uncertain':
                    rows.append(dict(run_id=r['run_id'],proposal=p['proposal'],candidate_id=p['candidate_id'],**result));counts[result['classification']]+=1;counts['fingerprint_complete']+=result['fingerprint_complete']
                    save('decision',rows[-1])
                else:
                    profile=_cached_profile(_json(b),key)
                    from experiments.ontology_review_recurrent import _semantic_graph
                    residual.append(dict(run_id=r['run_id'],proposal=p['proposal'],candidate_id=p['candidate_id'],path=str(child),parent_path=str(parent),errors=profile['errors'],mechanism=_semantic_graph(profile['mechanism']),fingerprint_complete=bool(result and result['fingerprint_complete'])))
                    errors.update(profile['errors'].values())
                    save('residual',residual[-1])
            except Exception as e:
                residual.append(dict(run_id=r['run_id'],proposal=p['proposal'],path=str(child),parent_path=str(parent),error=str(e)));errors[str(e)]+=1
                save('residual',residual[-1])
            n=len(rows)+len(residual)
            if n%100==0:
                print(key,n,dict(counts),'unmatched',len(residual),flush=True)
                (out/(key+'-'+args.suffix+'-progress.json')).write_text(json.dumps({'processed':n,'classified':len(rows),'unmatched':len(residual),'counts':dict(counts)},indent=2))
        (out/(key+'-'+args.suffix+'-decisions.json')).write_text(json.dumps(rows,indent=2))
        (out/(key+'-'+args.suffix+'-residuals.json')).write_text(json.dumps(residual,indent=2))
        summary={'classified':len(rows),'unmatched':len(residual),'counts':dict(counts),'errors':errors.most_common(),'source_rule_hashes':hashes,'source_rules_unchanged':hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in engine_files}}
        (out/(key+'-'+args.suffix+'-coverage.json')).write_text(json.dumps(summary,indent=2));print(key,'done',len(rows),'classified',len(residual),'unmatched',flush=True)

if __name__=='__main__':main()
