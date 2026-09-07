"""Second-pass review of parent-child execution shapes; no candidate execution."""
import ast
import collections
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from experiments.review_ontology_sources import clean_ast, dotted, execution_shape, read_sources


def graph_job(job):
    sha, sources = job
    definitions = {}
    trees = []
    try:
        for filename, text in sources.items():
            tree = ast.parse(text)
            trees.append((filename, tree))
            definitions.update({n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))})
    except SyntaxError:
        return sha, None
    classes = {name for name, n in definitions.items() if isinstance(n, ast.ClassDef)
               and any(dotted(b).endswith('.Module') for b in n.bases)}
    for _ in range(3):
        classes.update(name for name,n in definitions.items() if isinstance(n, ast.ClassDef)
                       and any(dotted(b) in classes for b in n.bases))
    builder = definitions.get('build_model')
    roots = ({dotted(n.func) for n in ast.walk(builder) if isinstance(n, ast.Call)} & classes) if builder else set()
    if not roots:
        roots = {name for name in ['GPT', 'TinyDecoderLM'] if name in classes}
    if not roots:
        referenced = {n.id for name in classes for n in ast.walk(definitions[name]) if isinstance(n,ast.Name) and n.id != name}
        roots = classes - referenced
    if len(roots) != 1:
        return sha, None
    reached = set(roots)
    for _ in range(12):
        previous = set(reached)
        for name in previous:
            reached.update(n.id for n in ast.walk(definitions[name]) if isinstance(n,ast.Name) and n.id in definitions)
        if reached == previous:
            break
    reached.update(name for name in ['encode_inputs','encode_targets','decode_targets','build_model'] if name in definitions)
    shape = execution_shape(definitions, reached, classes)
    used = {n.id for name in reached for n in ast.walk(definitions[name]) if isinstance(n,ast.Name)}
    globals_used = []
    for filename, tree in trees:
        for n in tree.body:
            if isinstance(n,(ast.Assign,ast.AnnAssign)):
                targets = n.targets if isinstance(n,ast.Assign) else [n.target]
                if any(isinstance(t,ast.Name) and t.id in used for t in targets):
                    globals_used.append(filename+clean_ast(n))
    return sha, hashlib.sha256((shape+json.dumps(globals_used)).encode()).hexdigest()


def adjudicate(output=Path('outputs/ontology')):
    snapshot = json.loads((output/'dashboard-snapshot.json').read_text(encoding='utf-8'))
    totals = {}
    for key,campaign in snapshot['campaigns'].items():
        file = output/(key+'.json')
        if not campaign.get('available') or not file.exists():
            continue
        doc = json.loads(file.read_text(encoding='utf-8'))
        points = {(r['run_id'],p['proposal']):(r,p) for r in campaign['runs'] for p in r['points']}
        paths = {}
        row_paths = {}
        for row in doc['rows']:
            run,point = points[(row['run_id'],row['proposal'])]
            root = Path(campaign['campaign'])/'runs'/run['run_id']
            path = root/(point.get('artifact_path') or 'candidates/'+row['candidate_id']).replace('\\','/')
            paths[str(path)] = path
            parents = [str(root/'candidates'/cid) for cid in point.get('parent_ids') or []]
            paths.update({parent:Path(parent) for parent in parents})
            row_paths[(row['run_id'],row['proposal'])] = (str(path),parents)
        print(key, 'checking execution shapes',len(paths),flush=True)
        bundles = {}
        source_ids = {}
        with ThreadPoolExecutor(max_workers=12) as pool:
            for path,sources in zip(paths,pool.map(read_sources,paths.values())):
                sha = hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
                bundles.setdefault(sha,sources)
                source_ids[path] = sha if sources else None
        shapes = {}
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i,(sha,shape) in enumerate(pool.map(graph_job,bundles.items(),chunksize=12)):
                shapes[sha] = shape
                if i and i%1000==0:
                    print(key,'checked',i,'/',len(bundles),flush=True)
        anchored = {}
        for row in doc['rows']:
            path,parents = row_paths[(row['run_id'],row['proposal'])]
            shape = shapes.get(source_ids[path])
            if row['family'] and shape and row['classification'] in {'preserving','changing'}:
                anchored.setdefault(shape,row['family'])
        refinements = 0
        for row in doc['rows']:
            path,parents = row_paths[(row['run_id'],row['proposal'])]
            sha = source_ids[path]
            shape = shapes.get(sha)
            row['execution_shape_sha256'] = shape
            if sha != row['source_sha256']:
                row.update(classification='unannotated',family=None,implemented=None)
                row['notes'] = 'Source changed or became unavailable after extraction; previous labels withheld.'
                continue
            # Removal of one existing position mechanism is not a new primitive.
            if row['classification']=='changing' and row.get('changed_components')==['position'] and len(parents)==1:
                parent_row = next((r for r in doc['rows'] if r['run_id']==row['run_id'] and r['candidate_id']==points[(row['run_id'],row['proposal'])][1]['parent_ids'][0]),None)
                before = parent_row and parent_row['fingerprint'].get('position')
                after = row['fingerprint'].get('position')
                if before and after and set(after.split(' + ')) < set(before.split(' + ')):
                    row.update(classification='uncertain',family=None)
                    row['notes'] = 'Removal of an already-present positional component is not counted as a new primitive. Whole-edit classification remains unresolved. '+row['notes']
            if row['classification']=='uncertain' and shape and len(parents)==1 and shape==shapes.get(source_ids[parents[0]]):
                row['classification'] = 'preserving'
                row['notes'] = 'Second pass: parent and child have the same rooted execution shape modulo standard affine/table/normalization coordinate wrappers. Input encoding, model factory and referenced global configuration are included. '+row['notes']
                refinements += 1
            if shape in anchored:
                row['family'] = anchored[shape]
        doc['rubric'] = 'source-review-v1 + execution-shape adjudication · partial source fingerprints; unresolved edits explicit'
        doc['adjudication'] = {'method':'rooted execution-shape comparison with factory, encoding and global configuration guards','new_preserving_labels':refinements}
        doc['review_code_sha256'] = {name:hashlib.sha256((Path('experiments')/name).read_bytes()).hexdigest() for name in ['review_ontology_sources.py','adjudicate_ontology_reviews.py']}
        # Replace only complete documents; the running dashboard never sees a partial write.
        temporary = file.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(doc,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
        temporary.replace(file)
        totals[key] = {'rows':len(doc['rows']),'classifications':dict(collections.Counter(r['classification'] for r in doc['rows'])),
                       'families':len({r['family'] for r in doc['rows'] if r['family']}),
                       'known_component_values':sum(v is not None for r in doc['rows'] for v in r['fingerprint'].values()),
                       'new_preserving_labels':refinements}
        print(key,totals[key],flush=True)
    (output/'review-summary.json').write_text(json.dumps(totals,indent=2),encoding='utf-8')


if __name__=='__main__':
    adjudicate()
