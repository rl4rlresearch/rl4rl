"""Read-only finite-corpus inventory for recurrent semantic review."""
import ast
import collections
import hashlib
import json
from pathlib import Path
from experiments.review_ontology_sources import read_sources, dotted
from experiments.ontology_semantics import program, SKIP_METHODS


def main():
    out=Path('outputs/ontology/semantic/recurrent');out.mkdir(parents=True,exist_ok=True)
    snapshot=json.loads(Path('outputs/ontology/v2/dashboard-snapshot-latest.json').read_text())
    for key in ['uci_har_pareto_v21','openevolve_v21_tiny_kws_rnn']:
        c=snapshot['campaigns'][key]
        rows=json.loads(Path('outputs/ontology/'+key+'.json').read_text())['rows']
        statuses={(r['run_id'],r['proposal']):r['classification'] for r in rows}
        paths={}
        for r in c['runs']:
            root=Path(c['campaign'])/'runs'/r['run_id']
            for p in r['points']:
                paths[str(root/(p.get('artifact_path') or 'candidates/'+p['candidate_id']))]={'run_id':r['run_id'],'proposal':p['proposal'],'status':statuses[(r['run_id'],p['proposal'])]}
        count=collections.Counter();classes=collections.Counter();methods=collections.Counter();seen=set();data=[]
        for i,(path,record) in enumerate(paths.items()):
            src=read_sources(Path(path));sha=hashlib.sha256(json.dumps(src,sort_keys=True).encode()).hexdigest()
            if sha in seen:continue
            seen.add(sha)
            try:
                defs,files,cls,reached,roots,env,globalenv,decisions,trees=program(src)
                signatures={}
                for name in reached:
                    if not isinstance(defs[name],ast.ClassDef):continue
                    classes[name]+=1
                    for m in defs[name].body:
                        if not isinstance(m,ast.FunctionDef) or m.name in SKIP_METHODS:continue
                        methods[m.name]+=1
                        ops=sorted(set(dotted(n.func) for n in ast.walk(m) if isinstance(n,ast.Call)))
                        count.update(ops)
                        signatures[name+'.'+m.name]={'source':ast.unparse(m),'file':files[name],'line':m.lineno,'operations':ops}
                data.append(dict(record,source_sha256=sha,path=path,methods=signatures,roots=sorted(roots),config=env))
            except Exception as e:data.append(dict(record,source_sha256=sha,path=path,error=str(e)))
            if i and i%500==0:print(key,i,flush=True)
        (out/(key+'-inventory.json')).write_text(json.dumps(data,indent=2))
        (out/(key+'-counts.json')).write_text(json.dumps({'sources':len(data),'operations':count.most_common(),'classes':classes.most_common(),'methods':methods.most_common()},indent=2))
        print(key,'done',len(data),flush=True)

if __name__=='__main__':main()
