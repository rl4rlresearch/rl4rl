"""Read-only coverage audit for explicitly reviewed full-source templates."""
import argparse
import collections
import json
from pathlib import Path

from experiments.audit_ontology_recurrent import read_sources
from experiments.ontology_review_recurrent import reviewed_template_key,source_sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);args=parser.parse_args()
    out=Path('outputs/ontology/semantic/recurrent')
    refs=[r for r in json.loads(Path('experiments/ontology_recurrent_references.json').read_text(encoding='utf-8'))['rows'] if r['campaign']==args.campaign and r.get('reviewed_template')]
    groups={}
    for ref in refs:
        spec=ref['reviewed_template'];key=json.dumps(spec['numeric_slots'],sort_keys=True)
        groups.setdefault(key,{'slots':spec['numeric_slots'],'templates':{}})['templates'][spec['whole_source_ast_sha256']]=ref
    inventory=json.loads((out/(args.campaign+'-inventory.json')).read_text(encoding='utf-8'))
    checkpoint=out/(args.campaign+'-template-coverage.jsonl');counts=collections.Counter();errors=collections.Counter()
    with checkpoint.open('w',encoding='utf-8') as handle:
        for index,record in enumerate(inventory):
            try:
                sources=read_sources(Path(record['path']))
                for group in groups.values():
                    match=reviewed_template_key(sources,group['slots'])
                    if not match or match[0] not in group['templates']:continue
                    ref=group['templates'][match[0]]
                    result={'run_id':record['run_id'],'proposal':record['proposal'],'source_sha256':source_sha(sources),'reference_run_id':ref['run_id'],'reference_proposal':ref['proposal'],'settings':match[1]}
                    counts[(ref['run_id'],ref['proposal'])]+=1;handle.write(json.dumps(result)+'\n');handle.flush();break
            except (OSError,ValueError,SyntaxError,TypeError) as exc:errors[type(exc).__name__+': '+str(exc)]+=1
            if (index+1)%250==0:print(args.campaign,index+1,'complete template matches',sum(counts.values()),flush=True)
    summary={'audited_sources':len(inventory),'complete_template_matches':sum(counts.values()),'by_reference':[{'run_id':key[0],'proposal':key[1],'matches':value} for key,value in counts.items()],'errors':dict(errors)}
    (out/(args.campaign+'-template-coverage.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8');print(summary,flush=True)


if __name__=='__main__':main()
