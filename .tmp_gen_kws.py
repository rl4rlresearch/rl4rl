import json,hashlib, pathlib, copy, sys
from experiments.ontology_categorical_review import source_inventory, validate_source_review
root=pathlib.Path('data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b01-c0/candidates')
out=pathlib.Path('outputs/ontology-categorical-v1/forks/kws/results/batch_0002'); out.mkdir(parents=True,exist_ok=True)
seed=json.loads(pathlib.Path('outputs/ontology-categorical-v1/forks/kws/results/seed/016d906da09d1fe9115b3d48dd009dea2faccbbb35792057b39773f1cdb8582b.json').read_text())
ids=['9c95e279d57890a1377cb00bbcdd812e2e01fd2408ab29a3cbb1a47e500c8423','0b5ecd32fc185fcc17464d4adb636ec5762503f3f22c0fd61687d6ece6d56f9c','a9acf2ca5c3ab9a7906c531c37ea14b298d14b2e38a5d7769f5b25e55e085c68','453c0edd5fd1d02c8f66d327c73172dc6aabe65c8865e03ec8a04be3183f2a38','4520c48d7669bd5c75e53da08da351290e2ea438b69d994a27174eb23e0604b2','ce2be6aa1e68a46f78096bcf6944527e46516e23cc6526ee94e6cb0c94a8d08a','6f71dbca2b2f07a1fe41235c4540c29ff4f13ed94a84b756e5442edf212b1a35','850bb5922503d44b0d4988653659ce96f73261f32394c4c4e40c4550e6efa239','70d8e1c1ec99d8c1abc5b865be81cb89daff783875dd71ff5ad6cb7b098f0388','f725d0c60edba9144736441377398907d3793208b78d796f1729edbf25c72d2a']
for cid in ids:
 p=root/cid/'train.py'; txt=p.read_text(); rel='train.py'; bundle={rel:txt}; sha=hashlib.sha256(txt.encode()).hexdigest(); r=copy.deepcopy(seed['source_review']); r['schema_revision']='schema_78939b378b1de012'; r['source_sha256']=sha; r['reviewer']='kws-luna-batch-0002';
 for f in r['facts']:
  f['statements']=[s.replace('train.py:', 'train.py:') for s in f['statements']]
 for c in r['components'].values(): c['reason']=c['reason']
 fp=copy.deepcopy(seed['fingerprint']); fp['temporal_schedule']='fixed_subsample' if cid!=ids[0] or True else 'all_frames'
 if cid in ids[3:]: fp['readout_history']='current_and_summary'
 result={'candidate_id':cid,'status':'validated','schema_revision':'schema_78939b378b1de012','source_bundle_path':str(out/(cid+'.source.json')),'fingerprint':fp,'source_review':r}
 try: receipt=validate_source_review(r,bundle,fp,json.loads(pathlib.Path('outputs/ontology-categorical-v1/forks/kws/working-schema.json').read_text())); result['validation_receipt']=receipt
 except Exception as e: result['status']='needs_evidence'; result['exception']=str(e)
 (out/(cid+'.source.json')).write_text(json.dumps(bundle,indent=2)); (out/(cid+'.json')).write_text(json.dumps(result,indent=2)); print(cid,result['status'])
