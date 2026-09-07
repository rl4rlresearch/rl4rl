"""Bounded review of deployment-time affine readout coordinate pruning.

Only a fully matched recurrent/core program and an unchanged readout feature
expression qualify. Parameter-derived column selection and reference-class
contrasts are permitted; input-derived selection and nonlinear heads are not.
"""
import ast
import copy
import json
from experiments.ontology_semantics import program, Specialize
from experiments.ontology_transition_review import settings_signature
from experiments.review_ontology_sources import dotted


def affine_output_features(value):
    """Prove fixed affine postprocessing of independent head outputs.

    Coordinate selections and fixed Helmert reconstruction may follow a head.
    A posterior-dependent selector, normalization by a signal, or nonlinear
    postprocessing cannot use this proof.
    """
    from experiments.ontology_review_recurrent import _readout_features,_json
    def scalar(n):
        if not isinstance(n,tuple):return isinstance(n,(str,int,float,bool,type(None)))
        return bool(n) and n[0] in {'number','literal','axes','Add','Sub','Mult','Div','Pow','USub','UAdd','sqrt','rsqrt'} and all(scalar(x) for x in n[1:])
    def fixed_index(n):
        if n is None:return True
        return isinstance(n,tuple) and bool(n) and n[0] in {'fixed-index','slice-index','axes-index'} and all(fixed_index(x) for x in n[1:] if isinstance(x,tuple)) and not any(isinstance(x,tuple) and x and x[0]=='computed-index' for x in n[1:])
    def extract(n):
        if not isinstance(n,tuple) or not n:return None
        if n[0]=='affine':return [_readout_features(n[1])]
        if n==('zero-logit',) or scalar(n):return []
        if n[0]=='select':
            index=n[2]
            if (isinstance(index,tuple) and len(index)==3 and index[0]=='axes-index'
                and index[1]==('slice-index',None,None,None) and fixed_index(index[2])):
                return extract(n[1])
            return None
        if n[0] in {'USub','UAdd'}:return extract(n[1])
        if n[0] in {'Add','Sub','combine'}:
            parts=[extract(x) for x in n[1:]]
            return None if any(p is None for p in parts) else [x for p in parts for x in p]
        if n[0] in {'Mult','Div'}:
            if scalar(n[2]) and n[2]!=('number',0):return extract(n[1])
            if n[0]=='Mult' and scalar(n[1]):return extract(n[2])
        return None
    parts=extract(value)
    if not parts:return None
    unique={_json(p):p for p in parts};features=[unique[k] for k in sorted(unique)]
    return features[0] if len(features)==1 else ('features',*features)


def coordinate_readout_signature(sources):
    # Local import avoids circularity; the main reviewer imports this helper only
    # after the equation interpreter is initialized.
    from experiments.ontology_review_recurrent import Equation, Unsupported, _readout_features
    defs,files,classes,reached,roots,envs,_,_,_=program(sources)
    root=next(iter(roots));node=defs[root]
    if len(roots)!=1:return None
    methods={m.name:m for m in node.body if isinstance(m,ast.FunctionDef)}
    if 'classify' not in methods:return None
    classifier=methods['classify'];ctor=methods.get('__init__')
    if not ctor:return None
    linear={dotted(n.targets[0]) for n in ast.walk(ctor) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and dotted(n.value.func).split('.')[-1]=='Linear'}
    called={dotted(n.func) for n in ast.walk(classifier) if isinstance(n,ast.Call)}
    heads=linear&called
    if not heads:return None
    # Any Linear also used by recurrent/core methods cannot be discarded.
    for name,m in methods.items():
        if name in {'__init__','classify','train'} or any(s in name.lower() for s in ['sync','deploy']):continue
        if any(isinstance(n,ast.Attribute) and dotted(n) in heads for n in ast.walk(m)):return None
    buffers={}
    for n in ast.walk(ctor):
        if isinstance(n,ast.Call) and dotted(n.func)=='self.register_buffer' and n.args and isinstance(n.args[0],ast.Constant):
            name='self.'+str(n.args[0].value)
            if any(isinstance(a,ast.Attribute) and dotted(a)==name for a in ast.walk(classifier)):buffers[name]=n
    # Buffer updates may depend on weights and sizes, never on observations.
    allowed_attrs=heads|set(buffers)|{'self.training'}
    excluded={'classify'}
    for name,m in methods.items():
        if name=='train' or any(s in name.lower() for s in ['sync','deploy']):
            attrs={dotted(n) for n in ast.walk(m) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self'}
            called_helpers={x for x in attrs if x[5:] in methods}
            if attrs-(allowed_attrs|called_helpers):return None
            if any(a.arg not in {'self','mode'} for a in m.args.args):return None
            excluded.add(name)
    # Every used buffer has static construction. An unknown setter means no proof.
    for attr in buffers:
        for name,m in methods.items():
            if name in excluded|{'__init__'}:continue
            if any(isinstance(n,ast.Attribute) and dotted(n)==attr for n in ast.walk(m)):return None
    modules={h:{'kind':'Linear'} for h in heads}
    class Readout(Equation):
        def expr(self,n):
            if isinstance(n,ast.Attribute) and dotted(n) in buffers:return ('coordinate-indices',)
            if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Tuple):
                axes=n.slice.elts
                # Pruning feature columns of an existing state/history tensor
                # is an affine coordinate edit. Whole-state tuple selections,
                # timestep/layer selections and changed observations remain.
                if len(axes)>=2 and isinstance(axes[-1],ast.Slice) and all(isinstance(x,ast.Slice) and x.lower is None and x.upper is None and x.step is None for x in axes[:-1]):
                    return self.expr(n.value)
            if isinstance(n,ast.Call):
                name=dotted(n.func);tail=name.split('.')[-1]
                if tail=='index_select' and isinstance(n.func,ast.Attribute):
                    indices=self.expr(n.args[1]) if len(n.args)>1 else None
                    if indices!=('coordinate-indices',):raise Unsupported('input-dependent readout indices')
                    return self.expr(n.func.value)
                if tail=='new_zeros':return ('zero-logit',)
            return super().expr(n)
    features=[]
    for training in [True,False]:
        config=dict(envs[root]);config['self.training']=training
        method=Specialize(config).visit(copy.deepcopy(classifier))
        reader=Readout(modules,methods,config)
        try:
            equation=reader.run(method)
            proved=affine_output_features(equation)
            value=proved if proved is not None else _readout_features(equation)
        except (Unsupported,ValueError,TypeError,IndexError):return None
        def drop_zero(n):
            if not isinstance(n,tuple):return n
            if n and n[0]=='features':
                xs=tuple(drop_zero(x) for x in n[1:] if x!=('zero-logit',))
                return xs[0] if len(xs)==1 else ('features',*xs)
            return (n[0],*(drop_zero(x) if isinstance(x,tuple) else x for x in n[1:]))
        features.append(drop_zero(value))
    # A contrast/pruned-coordinate head remains affine in the same source feature
    # collection. It does not authorize adding a new mean/max/state observation.
    if features[0]!=features[1]:return None
    class RemoveHead(ast.NodeTransformer):
        def visit_ClassDef(self,n):
            # Other reachable cell/helper classes remain in the exact core
            # comparison. A same-named attribute in another class is not this
            # root model's affine head.
            return self.generic_visit(n) if n.name==root else n
        def visit_FunctionDef(self,n):
            if n.name in excluded:return None
            if n.name=='__init__':
                kept=[]
                for st in n.body:
                    attrs={dotted(x) for x in ast.walk(st) if isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name) and x.value.id=='self'}
                    if attrs and attrs<=allowed_attrs:continue
                    if isinstance(st,ast.Expr) and isinstance(st.value,ast.Call) and dotted(st.value.func)=='self.register_buffer' and st.value.args and 'self.'+str(getattr(st.value.args[0],'value',None)) in buffers:continue
                    if attrs and all(x[5:] in excluded for x in attrs):continue
                    kept.append(st)
                n.body=kept or [ast.Pass()]
            return self.generic_visit(n)
    stripped={f:ast.unparse(ast.fix_missing_locations(RemoveHead().visit(ast.parse(src)))) for f,src in sources.items()}
    return settings_signature(stripped),features[0]


def coordinate_pruning_contract(sources):
    """Describe a source-checked deployment-only affine coordinate pruner.

    The general recurrent interpreter intentionally does not execute a model's
    ``train(False)`` synchronization hook.  This narrow contract closes that
    gap only when :func:`coordinate_readout_signature` has already proved that
    both training and inference heads consume the same readout features and
    every buffer used at inference is a registered coordinate selector.  The
    hook may inspect learned head weights, but cannot receive observations or
    update model state other than the selected-coordinate buffer and its
    auxiliary affine head.  Random selector construction is rejected.
    """
    signature = coordinate_readout_signature(sources)
    if signature is None:
        return None
    defs, _, classes, reached, roots, _, _, _, _ = program(sources)
    if len(roots) != 1:
        return None
    root = next(iter(roots))
    if root not in reached or root not in classes:
        return None
    methods = {
        method.name: method for method in defs[root].body
        if isinstance(method, ast.FunctionDef)
    }
    ctor = methods.get('__init__')
    classifier = methods.get('classify')
    if ctor is None or classifier is None:
        return None
    linear = {
        dotted(node.targets[0]) for node in ast.walk(ctor)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        and dotted(node.value.func).split('.')[-1] == 'Linear'
    }
    called = {dotted(node.func) for node in ast.walk(classifier) if isinstance(node, ast.Call)}
    heads = linear & called
    buffers = {
        'self.' + str(node.args[0].value)
        for node in ast.walk(ctor)
        if isinstance(node, ast.Call) and dotted(node.func) == 'self.register_buffer'
        and node.args and isinstance(node.args[0], ast.Constant)
        and any(isinstance(use, ast.Attribute) and dotted(use) == 'self.' + str(node.args[0].value)
                for use in ast.walk(classifier))
    }
    if not heads or not buffers:
        return None
    # A synchronization hook may be called ``train`` or named for deployment,
    # but it has no tensor argument: source observations cannot influence its
    # coordinate selection.  Its self attributes are restricted to the already
    # proved heads/buffers, and it may not introduce stochastic selection.
    hooks = set()
    random_calls = {
        'rand', 'rand_like', 'randn', 'randn_like', 'randint', 'bernoulli',
        'multinomial', 'normal', 'poisson', 'random',
    }
    allowed = heads | buffers | {'self.training'}
    for name, method in methods.items():
        if name != 'train' and not any(part in name.lower() for part in ('sync', 'deploy', 'inference')):
            continue
        attrs = {
            dotted(node) for node in ast.walk(method)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == 'self'
        }
        if not attrs or attrs - allowed:
            return None
        if any(argument.arg not in {'self', 'mode'} for argument in method.args.args):
            return None
        if any(
            dotted(node.func).split('.')[-1] in random_calls
            for node in ast.walk(method) if isinstance(node, ast.Call)
        ):
            return None
        hooks.add(name)
    if not hooks:
        return None
    return {
        'signature': signature[0],
        'features': signature[1],
        'buffers': tuple(sorted(buffers)),
        'hooks': tuple(sorted(hooks)),
    }


def same_coordinate_readout(before,after):
    try:
        a,b=coordinate_readout_signature(before),coordinate_readout_signature(after)
        return bool(a and a==b)
    except (ValueError,TypeError,SyntaxError,IndexError,RecursionError):return False


def frequency_pool_extent_signature(sources):
    """Existing contiguous adjacent-frequency pair averaging: extent is a setting.

    Require the complete helper equation, including normalization before pooling,
    the frequency axis, adjacent pairs, and untouched prefix concatenation. This
    does not erase a first pooling transform, a different axis or signal-selected
    region, and the rest of the source must independently match.
    """
    from experiments.ontology_semantics import constant,UNKNOWN
    defs,files,classes,reached,roots,envs,_,_,_=program(sources)
    if len(roots)!=1:return None
    root=next(iter(roots));node=defs[root];env=envs[root]
    methods={m.name:m for m in node.body if isinstance(m,ast.FunctionDef)}
    ctor=methods.get('__init__')
    if ctor is None:return None
    normalizers={}
    for n in ast.walk(ctor):
        if not isinstance(n,ast.Assign) or len(n.targets)!=1 or not isinstance(n.value,ast.Call):continue
        if dotted(n.value.func).split('.')[-1]!='LayerNorm':continue
        value=constant(n.value.args[0],env) if n.value.args else UNKNOWN
        if type(value) is int:normalizers[dotted(n.targets[0])]=value
    matches={}
    def frequency_slice(n,name,lower,upper):
        if not isinstance(n,ast.Subscript) or not isinstance(n.value,ast.Name) or n.value.id!=name:return False
        if not isinstance(n.slice,ast.Tuple) or len(n.slice.elts)!=2:return False
        first,last=n.slice.elts
        if not isinstance(first,ast.Constant) or first.value is not Ellipsis or not isinstance(last,ast.Slice) or last.step is not None:return False
        return (last.lower is None if lower is None else constant(last.lower,env)==lower) and (last.upper is None if upper is None else constant(last.upper,env)==upper)
    def last_axis(call):
        return len(call.keywords)==1 and call.keywords[0].arg=='dim' and constant(call.keywords[0].value,env)==-1
    for name,m in methods.items():
        body=[n for n in m.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
        if len(body)!=3 or not all(isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) for n in body[:2]) or not isinstance(body[2],ast.Return):continue
        norm,pool=body[:2];norm_name=norm.targets[0].id;pool_name=pool.targets[0].id
        call=norm.value
        if not isinstance(call,ast.Call) or dotted(call.func) not in normalizers or len(call.args)!=1 or call.keywords:continue
        args=[a.arg for a in m.args.args if a.arg!='self']
        if len(args)!=1 or not isinstance(call.args[0],ast.Name) or call.args[0].id!=args[0]:continue
        mean=pool.value
        if not isinstance(mean,ast.Call) or not isinstance(mean.func,ast.Attribute) or mean.func.attr!='mean' or mean.args or not last_axis(mean):continue
        reshape=mean.func.value
        if not isinstance(reshape,ast.Call) or not isinstance(reshape.func,ast.Attribute) or reshape.func.attr!='reshape' or len(reshape.args)!=3 or reshape.keywords:continue
        if not isinstance(reshape.args[0],ast.Starred):continue
        expected=ast.parse(norm_name+'.shape[:-1]',mode='eval').body
        if ast.dump(reshape.args[0].value,include_attributes=False)!=ast.dump(expected,include_attributes=False):continue
        pairs=constant(reshape.args[1],env);pair_width=constant(reshape.args[2],env)
        if type(pairs) is not int or pairs<=0 or pair_width!=2:continue
        start=normalizers[dotted(call.func)]-2*pairs
        if start<0 or not frequency_slice(reshape.func.value,norm_name,start,None):continue
        concat=body[2].value
        if not isinstance(concat,ast.Call) or dotted(concat.func) not in {'torch.cat','torch.concat','torch.concatenate'} or len(concat.args)!=1 or not last_axis(concat):continue
        pieces=concat.args[0]
        if not isinstance(pieces,(ast.Tuple,ast.List)) or len(pieces.elts)!=2:continue
        if not frequency_slice(pieces.elts[0],norm_name,None,start) or not isinstance(pieces.elts[1],ast.Name) or pieces.elts[1].id!=pool_name:continue
        replacement=copy.deepcopy(m)
        for n in ast.walk(replacement):
            if frequency_slice(n,norm_name,start,None):n.slice.elts[1].lower=ast.Constant(value=2)
            elif frequency_slice(n,norm_name,None,start):n.slice.elts[1].upper=ast.Constant(value=2)
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='reshape':n.args[1]=ast.Constant(value=1)
        matches[name]=replacement
    if not matches:return None
    class CanonicalExtent(ast.NodeTransformer):
        def visit_ClassDef(self,n):return self.generic_visit(n) if n.name==root else n
        def visit_FunctionDef(self,n):return matches.get(n.name,n)
    stripped={f:ast.unparse(ast.fix_missing_locations(CanonicalExtent().visit(ast.parse(src)))) for f,src in sources.items()}
    return tuple(sorted(matches)),settings_signature(stripped)


def same_frequency_pool_extent(before,after):
    if not all(any('reshape' in source and 'mean' in source for source in bundle.values()) for bundle in [before,after]):return False
    try:
        a,b=frequency_pool_extent_signature(before),frequency_pool_extent_signature(after)
        return bool(a and b and a[1] and a==b)
    except (ValueError,TypeError,SyntaxError,IndexError,RecursionError):return False


def latent_split_width_signature(sources):
    """Only coordinated widths around a learned latent feature split may vary."""
    from experiments.ontology_review_recurrent import module_call_name
    trees={name:ast.parse(source) for name,source in sources.items()}
    defs,files,classes,reached,roots,envs,_,_,_=program(sources)
    if len(roots)!=1:return None
    root=next(iter(roots));tree=trees[files[root]]
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==root)
    ctor=next((n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'),None)
    if ctor is None:return None
    modules={}
    def register(name,call):
        if (isinstance(call,ast.Call) and dotted(call.func).split('.')[-1]=='Linear' and len(call.args)==2
            and all(isinstance(n,ast.Constant) and type(n.value) is int and n.value>1 for n in call.args)):
            modules[name]=call
    for n in ast.walk(ctor):
        if not isinstance(n,ast.Assign) or len(n.targets)!=1 or not isinstance(n.value,ast.Call):continue
        target=dotted(n.targets[0]);register(target,n.value)
        if dotted(n.value.func).split('.')[-1]=='ModuleList' and len(n.value.args)==1 and isinstance(n.value.args[0],(ast.List,ast.Tuple)):
            for index,call in enumerate(n.value.args[0].elts):register(target+'['+str(index)+']',call)
    accepted=[];allowed_calls=set();touched=set()
    for method in cls.body:
        if not isinstance(method,ast.FunctionDef) or method.name=='__init__':continue
        parents={id(child):node for node in ast.walk(method) for child in ast.iter_child_nodes(node)}
        for assign in ast.walk(method):
            if not isinstance(assign,ast.Assign) or len(assign.targets)!=1 or not isinstance(assign.targets[0],(ast.Tuple,ast.List)):continue
            split=assign.value
            if not isinstance(split,ast.Call) or dotted(split.func)!='torch.split' or len(split.args)!=2 or len(split.keywords)!=1 or split.keywords[0].arg!='dim' or not isinstance(split.keywords[0].value,ast.Constant) or split.keywords[0].value.value!=1:continue
            widths=split.args[1];targets=assign.targets[0].elts
            if not isinstance(widths,(ast.Tuple,ast.List)) or len(widths.elts)!=len(targets) or len(targets)<2:continue
            if not all(isinstance(x,ast.Name) for x in targets) or not all(isinstance(x,ast.Constant) and type(x.value) is int and x.value>1 for x in widths.elts):continue
            sizes=[x.value for x in widths.elts]
            projection=split.args[0]
            while isinstance(projection,ast.Call) and dotted(projection.func) in {'torch.tanh','torch.sigmoid','torch.relu','F.relu','F.gelu','F.silu'} and len(projection.args)==1 and not projection.keywords:projection=projection.args[0]
            if not isinstance(projection,ast.Call) or len(projection.args)!=1 or projection.keywords:continue
            upstream=module_call_name(projection.func,{})
            if upstream not in modules or modules[upstream].args[1].value!=sum(sizes):continue
            downstream=[];calls={id(projection)};valid=True
            for target,width in zip(targets,sizes):
                stores=[n for n in ast.walk(method) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id==target.id]
                uses=[n for n in ast.walk(method) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id==target.id]
                if len(stores)!=1 or not uses:valid=False;break
                for use in uses:
                    call=parents[id(use)]
                    if not isinstance(call,ast.Call) or len(call.args)!=1 or call.args[0] is not use or call.keywords:valid=False;break
                    name=module_call_name(call.func,{})
                    if name not in modules or modules[name].args[0].value!=width:valid=False;break
                    downstream.append(name);calls.add(id(call))
                if not valid:break
            if valid:
                accepted.append((split,upstream,downstream));allowed_calls.update(calls);touched.update([upstream,*downstream])
    if not accepted:return None
    # A dimension slot may not also feed another unaccounted invocation.
    for method in cls.body:
        if not isinstance(method,ast.FunctionDef) or method.name=='__init__':continue
        for n in ast.walk(method):
            if isinstance(n,ast.Call) and module_call_name(n.func,{}) in touched and id(n) not in allowed_calls:return None
    for split,upstream,downstream in accepted:
        modules[upstream].args[1]=ast.Constant(value='reviewed latent total width')
        for name in downstream:modules[name].args[0]=ast.Constant(value='reviewed latent part width')
        for index in range(len(split.args[1].elts)):split.args[1].elts[index]=ast.Constant(value='reviewed latent part width')
    class StripDocs(ast.NodeTransformer):
        def visit_Expr(self,node):return None if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str) else self.generic_visit(node)
    return json.dumps({name:ast.dump(StripDocs().visit(tree),include_attributes=False) for name,tree in trees.items()},sort_keys=True)


def same_latent_split_widths(before,after):
    if not all(any('split' in source for source in bundle.values()) for bundle in [before,after]):return False
    try:
        a,b=latent_split_width_signature(before),latent_split_width_signature(after)
        return bool(a and a==b)
    except (ValueError,TypeError,SyntaxError,IndexError,RecursionError):return False
