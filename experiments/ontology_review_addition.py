"""Addition-specific source adjudication with explicit signal/parameter separation.

Candidate code is parsed, never imported or executed.  A preserving result needs
a matched rooted signal computation after the documented affine-coordinate
quotient.  Unknown operations/control flow are retained, not guessed away.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from experiments.ontology_semantics import program, constant, UNKNOWN
from experiments.ontology_seed_rubric import seed_fingerprint
from experiments.ontology_transition_review import NumericSettings
from experiments.review_ontology_sources import CORE, TASK_KEYS, dotted
from experiments.ontology_review_addition_encodings import encoding_descriptor

VERSION = 'addition-signal-review-v1'
AFFINE = {'Linear', 'linear'}
LOOKUP = {'Embedding', 'embedding'}
NORM = {'LayerNorm', 'RMSNorm', 'layer_norm', 'rms_norm'}
ACT = {'gelu','relu','silu','sigmoid','tanh','leaky_relu','GELU','ReLU','SiLU','Sigmoid','Tanh','LeakyReLU'}
LAYOUT = {'view','reshape','contiguous','flatten','unflatten','unsqueeze','squeeze','to','float','double','type_as','clone','detach'}
PARAM_OPS = LAYOUT | {'cat','stack','pad','zeros','zeros_like','ones','ones_like','empty','empty_like','full','full_like','new_zeros','new_ones','new_empty','new_full','sum','mean','repeat','repeat_interleave','expand','expand_as','transpose','permute','T','chunk','split','narrow','select','index_select','index_copy','index_copy_','scatter','scatter_','scatter_add','scatter_add_','tensor','as_tensor','arange','tril','triu','eye','diag','diagonal','diag_embed','copy_','fill_','zero_','add_','sub_','mul_','div_','item','size','numel','dim','len','int','float','bool','max','min','clamp','clamp_min','clamp_max','abs','sqrt','square','rsqrt','norm','linalg.vector_norm','linalg.qr','linalg.solve','linalg.inv','matmul','einsum','bmm','mm','addmm','index_put','index_put_','getattr','hasattr'}
INIT_METHODS = {'__init__','_init_weights','_initialize_weights','init_weights','reset_parameters','reset_from_full','initialize_from_full_normal','set_initial_weight','_set_from_full','_set_weight','_set_rest','set_from_full','initialize_reference'}
PARAM_OPS |= {'masked_fill','where','minimum','maximum','softmax','log_softmax','tril_indices','triu_indices','finfo','iinfo'}
PARAM_OPS |= {'range','enumerate','zip','list','tuple','unbind','outer','dot','t','gather','masked_scatter','masked_scatter_','add','sub','mul','div','zero','logsumexp','rfft','fft','append','extend'}
PARAM_OPS |= {'sorted','sinc','normalize','one_hot','flip','remainder','lt','le','gt','ge','eq','ne','logical_and','logical_or','logical_not','long','int','tolist','irfft','all','any','nonzero','vector_norm','view_as','prod'}
PARAM_OPS.add('complex') # Independent real/imaginary coordinates; signal use remains unsupported.


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class Val:
    key: tuple
    signal: bool = False
    kind: str = 'parameter'


def learned(value):
    def has(key):return key==('parameter',) or isinstance(key,tuple) and any(has(child) for child in key)
    return value.kind=='parameter' or value.kind=='sequence' and has(value.key)


P = Val(('parameter',))
FIXED = Val(('fixed coefficient',),kind='fixed')
META = Val(('shape/configuration',),kind='metadata')
INP = Val(('input',),True,'signal')


def node(op, *args):
    return Val((op,)+tuple(a.key if isinstance(a,Val) else a for a in args), any(a.signal for a in args if isinstance(a,Val)), 'signal' if any(a.signal for a in args if isinstance(a,Val)) else 'parameter')


def affine_input(v):
    key=v.key
    while key:
        if key[0] in {'affine','offset','scale'}:key=key[1];continue
        if key[0]=='MatMult' and key[2] in {('parameter',),('fixed coefficient',)}:key=key[1];continue
        break
    if key and key[0]=='index' and final_feature_slice(key[2]):key=key[1]
    return Val(key,v.signal,v.kind)


@lru_cache(maxsize=256)
def final_feature_slice(dump):
    """Validate exactly [..., slice] without executing the serialized AST."""
    try:
        outer=ast.parse(dump,mode='eval').body
        if not isinstance(outer,ast.Call) or dotted(outer.func)!='Tuple':return False
        elements=next(k.value for k in outer.keywords if k.arg=='elts')
        if not isinstance(elements,ast.List) or len(elements.elts)!=2:return False
        first,last=elements.elts
        return isinstance(first,ast.Call) and dotted(first.func)=='Constant' and any(k.arg=='value' and isinstance(k.value,ast.Name) and k.value.id=='Ellipsis' for k in first.keywords) and isinstance(last,ast.Call) and dotted(last.func)=='Slice'
    except (SyntaxError,StopIteration,TypeError):return False


def lookup_origin(key):
    """Linear output coordinates of one unchanged lookup stay a lookup.

    Two digit lookups at different indices are deliberately not merged: that
    is the compositional representation boundary used by this campaign.
    """
    if not isinstance(key,tuple) or not key:return None
    if key[0]=='lookup':return key[1]
    if key[0] in {'affine','offset','scale'}:return lookup_origin(key[1])
    if key[0]=='MatMult':
        found=[lookup_origin(k) for k in key[1:]]
        return next((x for x in found if x is not None),None) if sum(x is not None for x in found)==1 else None
    return None


def coordinate_signal_key(key,in_lookup_index=False):
    """Quotient source-proven independent feature scales/biases for comparison.

    Their source, positional score geometry and parameter generators remain in
    the descriptive profile. Products of two input-dependent tensors, casts,
    sequence indices and left-side matrix mixing remain explicit.
    """
    if not isinstance(key,tuple) or not key:return key
    value=tuple(coordinate_signal_key(v,in_lookup_index or key[0]=='lookup') for v in key)
    if value[0]=='lookup':
        origin=categorical_lookup_origin(value)
        if origin is not None:return ('lookup',origin)
    if value[0] in {'offset','scale'} or value[0]=='normalize' and not in_lookup_index:return value[1]
    if not in_lookup_index and value[0]=='Sub' and len(value)==3:
        mean=value[2]
        if isinstance(mean,tuple) and len(mean)==3 and mean[0]=='mean' and mean[1]==value[1] and isinstance(mean[-1],tuple):
            options=dict(mean[-1]) if all(isinstance(x,tuple) and len(x)==2 for x in mean[-1]) else {}
            if options.get('dim') in {'UnaryOp(op=USub(), operand=Constant(value=1))','Constant(value=-1)'} and options.get('keepdim')=='Constant(value=True)':return value[1]
    if value[0]=='affine':
        base=value[1]
        while isinstance(base,tuple) and base and base[0]=='affine':base=base[1]
        return ('affine',base)
    return value


def categorical_lookup_origin(key):
    """Disjoint token-table storage is still one categorical lookup.

    Decimal decomposition, products/sums of independent digit embeddings and
    arbitrary signal transforms are excluded; those define new representations.
    """
    if not isinstance(key,tuple) or not key:return None
    if key[0] in {'affine','offset','scale'}:return categorical_lookup_origin(key[1])
    if key[0]!='lookup':return None
    def origin(index):
        if index==():return ()
        if not isinstance(index,tuple):return None
        if index[0] in {'input','role input'}:return index
        if index[0] in {'literal','shape/configuration','parameter','fixed coefficient'}:return ()
        if index[0] in {'offset','scale','clamp','clamp_min','clamp_max','where','lt','le','gt','ge','eq','ne','BitAnd','BitOr','tuple'} or str(index[0]).startswith('comparison:'):
            live=[]
            for child in index[1:]:
                if not isinstance(child,tuple):continue
                value=origin(child)
                if value is None:return None
                if value:live.append(value)
            return live[0] if live and all(v==live[0] for v in live) else () if not live else None
        # Keyword metadata (e.g. clamp min/max) cannot introduce signal.
        if all(isinstance(v,tuple) and len(v)==2 and isinstance(v[1],str) and ('Constant(' in v[1] or 'UnaryOp(' in v[1]) for v in index):return ()
        return None
    return origin(key[1]) or None


def feature_affine(key,anchor):
    """An affine map of the same final-axis features, with no time selection."""
    if key==anchor:return True
    if not isinstance(key,tuple) or not key:return False
    op=key[0]
    if op in {'parameter','fixed coefficient','literal','shape/configuration'}:return True
    if op in {'affine','scale','offset','USub','UAdd'}:return feature_affine(key[1],anchor)
    if op in {'Add','Sub'}:return all(feature_affine(x,anchor) for x in key[1:])
    if op=='MatMult':
        def depends(x):return x==anchor or isinstance(x,tuple) and any(depends(y) for y in x)
        # Only multiplication on the right maps final-axis features. A fixed
        # coefficient matrix on the left can mix token/history positions.
        return depends(key[1]) and not depends(key[2]) and all(feature_affine(x,anchor) for x in key[1:])
    if op=='index':
        # Ellipsis + a final slice is a feature-coordinate restriction.  A
        # positional/time index is deliberately not covered by this rule.
        return final_feature_slice(key[2]) and feature_affine(key[1],anchor)
    if op=='cat':
        args=list(key[1:]);kw=args.pop() if args and isinstance(args[-1],tuple) else ()
        if kw not in {(('dim', 'UnaryOp(op=USub(), operand=Constant(value=1))'),),(('dim','Constant(value=-1)'),)}:return False
        return all(feature_affine(x,anchor) for x in args)
    if op=='tuple':return all(feature_affine(x,anchor) for x in key[1:])
    return False


def formatting_family(name,method):
    """Identify audited decimal formatting; retain unknown encodings exactly.

    Token-ID offsets and delimiter/EOS IDs are coordinates of a token alphabet,
    not a new operand representation. Radix, operand algebra and digit order
    remain part of the family description.
    """
    calls={dotted(n.func).split('.')[-1] for n in ast.walk(method) if isinstance(n,ast.Call)}
    if any(isinstance(n,(ast.If,ast.For,ast.While,ast.IfExp)) for n in ast.walk(method)):return None
    if calls-{'cat','full_like','zeros_like','ones_like'}:return None
    args=[a.arg for a in method.args.args]
    def trace_return(marked,initial):
        bindings=dict(initial)
        def trace(n):
            if id(n) in marked:return ('data',)
            if isinstance(n,ast.Name):return bindings.get(n.id,('unknown',))
            if isinstance(n,ast.Call) and dotted(n.func).split('.')[-1] in {'full_like','zeros_like','ones_like'}:return ()
            if isinstance(n,ast.Call) and dotted(n.func).split('.')[-1]=='cat' and n.args and isinstance(n.args[0],(ast.Tuple,ast.List)):
                return sum((trace(x) for x in n.args[0].elts),())
            return ('unknown',)
        for n in method.body:
            if isinstance(n,ast.Assign):
                for target in n.targets:
                    if isinstance(target,ast.Name):bindings[target.id]=trace(n.value)
            if isinstance(n,ast.Return):return trace(n.value)==('data',)
        return False
    if name=='encode_inputs' and len(args)>=2:
        # Audit the pair expression itself, not a suggestive variable name.
        pairs=[]
        for n in ast.walk(method):
            if not isinstance(n,ast.BinOp) or not isinstance(n.op,ast.Add):continue
            if isinstance(n.left,ast.BinOp) and isinstance(n.left.op,ast.Mult) and isinstance(n.left.left,ast.Name) and n.left.left.id==args[0] and isinstance(n.left.right,ast.Constant) and n.left.right.value==10 and isinstance(n.right,ast.Name) and n.right.id==args[1]:pairs.append(n)
            if isinstance(n.left,ast.BinOp) and isinstance(n.left.op,ast.Add) and isinstance(n.left.left,ast.Constant):
                core=n.left.right
                if isinstance(core,ast.BinOp) and isinstance(core.op,ast.Mult) and isinstance(core.left,ast.Name) and core.left.id==args[0] and isinstance(core.right,ast.Constant) and core.right.value==10 and isinstance(n.right,ast.Name) and n.right.id==args[1]:pairs.append(n)
        if not pairs:return None
        # No extra signal arithmetic may be concealed beside the recognized
        # expression (e.g. digit reversal, carry extraction or answer code).
        allowed=set()
        for p in pairs:allowed.update(id(n) for n in ast.walk(p))
        for n in ast.walk(method):
            if isinstance(n,ast.BinOp) and id(n) not in allowed:
                if isinstance(n.op,ast.Add) and (isinstance(n.left,ast.Constant) or isinstance(n.right,ast.Constant)):continue
                return None
        if not trace_return({id(p) for p in pairs},{}):return None
        return 'ordered base-10 operand-pair token sequence; unchanged incoming digit order'
    if name=='encode_targets' and len(args)==1:
        if any(isinstance(n,(ast.BinOp,ast.UnaryOp)) for n in ast.walk(method)):return None
        return 'supplied decimal digit targets; optional constant end token' if trace_return(set(),{args[0]:('data',)}) else None
    return None


def target_ast_key(method):
    node=copy.deepcopy(method)
    if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def target_references():
    rows=json.loads(Path(__file__).with_name('ontology_addition_pair_references.json').read_text(encoding='utf-8')).get('target_encodings',[])
    result={}
    for row in rows:
        if target_ast_key(ast.parse(row['source']).body[0])!=row['ast_sha256']:raise ValueError('Target encoding reference binding changed')
        result[row['ast_sha256']]=row['descriptor']
    return result


def target_descriptor(method,sources):
    descriptor=target_references().get(target_ast_key(method))
    if descriptor is None:return None
    # All reviewed functions are closed over their sole tensor argument and
    # ordinary torch operations. Do not reuse them after a global rebinding.
    if any(isinstance(n,ast.Name) and n.id=='torch' for n in ast.walk(method)):
        bound=False
        for source in sources.values():
            tree=ast.parse(source)
            if any(isinstance(n,ast.FunctionDef) and n.name==method.name and target_ast_key(n)==target_ast_key(method) for n in tree.body):
                bound=any(isinstance(n,ast.Import) and any(a.name=='torch' and a.asname in {None,'torch'} for a in n.names) for n in tree.body)
                for n in ast.walk(tree):
                    targets=n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,(ast.AnnAssign,ast.AugAssign,ast.NamedExpr)) else []
                    if any(dotted(t)=='torch' or dotted(t).startswith('torch.') for t in targets):return None
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name=='torch':return None
                    if isinstance(n,ast.ImportFrom) and any((a.asname or a.name)=='torch' for a in n.names):return None
                    if isinstance(n,ast.Import) and any((a.asname or a.name)=='torch' and a.name!='torch' for a in n.names):return None
        if not bound:return None
    return copy.deepcopy(descriptor)


class Unsupported(ValueError):
    pass


class SignalReview:
    """Symbolically retain the input computation while isolating coefficients.

    This is deliberately narrower than Python: mutation, runtime unknown
    branches and arbitrary callables remain explicit and prevent full coverage.
    """
    def __init__(self, sources):
        self.sources=sources
        self.defs,self.files,self.classes,self.reached,self.roots,self.class_envs,self.env,self.decisions,self.trees=program(sources)
        # The shared resolver knows nn.Module subclasses. These campaigns also
        # specialize nn.Linear/Embedding directly, so retain those subclasses.
        self.classes |= {c for c in self.reached if isinstance(self.defs[c],ast.ClassDef) and any(dotted(b).split('.')[-1] in AFFINE|LOOKUP|NORM for b in self.defs[c].bases)}
        for _ in range(len(self.defs)):
            self.classes |= {c for c in self.reached if isinstance(self.defs[c],ast.ClassDef) and any(dotted(b) in self.classes for b in self.defs[c].bases)}
        self.methods={c:{m.name:m for m in self.defs[c].body if isinstance(m,ast.FunctionDef)} for c in self.reached&self.classes}
        for c in self.methods:self.class_envs.setdefault(c,dict(self.env))
        self.attrs={c:{} for c in self.methods}
        self.modules={c:{} for c in self.methods}
        self.buffers={c:set() for c in self.methods}
        self.effects=set();self.parameter_ops=set();self.signal_ops=set();self.evidence=[];self.failures=[]
        self.roles={}
        self.role_stack=[]
        self.syntax_stack=[];self.score_forms=[]
        self.geometry_forms=[]
        self.position_forms=[]
        self.score_offsets=[]
        self.score_offset_geometry=[]
        self.buffer_lookups=set()
        self.output_forms=set()
        self.initialization_aliases=[]
        self.kernel_mechanisms=set()
        self.parameter_nonlinear_sites=[]
        self.parameter_product_sites=[]
        self.additive_buffers=set()
        self.stack=[]
        self.functional_aliases={a.asname or a.name:a.name for tree in self.trees.values() for imp in tree.body if isinstance(imp,ast.ImportFrom) and imp.module=='torch.nn.functional' for a in imp.names}
        for c,ms in self.methods.items():
            for method in ms.values():
                for n in ast.walk(method):
                    if isinstance(n,ast.Call) and dotted(n.func)=='self.register_buffer' and n.args and isinstance(n.args[0],ast.Constant):
                        self.buffers[c].add('self.'+str(n.args[0].value))
            init=ms.get('__init__')
            if init:
                for n in ast.walk(init):
                    if isinstance(n,(ast.Assign,ast.AnnAssign)):
                        for t in n.targets if isinstance(n,ast.Assign) else [n.target]:
                            name=dotted(t)
                            if name.startswith('self.'):
                                self.attrs[c][name]=n.value
                                self.modules[c].pop(name,None)
                                if isinstance(n.value,ast.Call):
                                    cal=dotted(n.value.func).split('.')[-1]
                                    if cal in AFFINE|LOOKUP|NORM|ACT|{'Dropout','Identity','Sequential','ModuleList','ModuleDict'}|self.classes:
                                        self.modules[c][name]=(cal,n.value)
                    if isinstance(n,ast.Call) and dotted(n.func)=='self.register_buffer' and n.args and isinstance(n.args[0],ast.Constant):
                        self.attrs[c]['self.'+str(n.args[0].value)]=n.args[1] if len(n.args)>1 else ast.Constant(None)
                        self.buffers[c].add('self.'+str(n.args[0].value))
        # Resolve shared module constructor arguments from actual call sites.
        # An annotation alone is not evidence of the instantiated module type.
        for _ in range(3):
            for c,ms in self.methods.items():
                for base in self.defs[c].bases:
                    name=dotted(base)
                    if name in self.methods:
                        self.buffers[c]|=self.buffers[name]
                        for attr,value in self.attrs[name].items():self.attrs[c].setdefault(attr,value)
                        for attr,value in self.modules[name].items():self.modules[c].setdefault(attr,value)
                init=ms.get('__init__')
                if not init:continue
                aliases={}
                for caller,modules in self.modules.items():
                    for _,container in list(modules.values()):
                        for call in ast.walk(container):
                            if not isinstance(call,ast.Call) or dotted(call.func).split('.')[-1]!=c:continue
                            bindings={arg.arg:value for arg,value in zip(init.args.args[1:],call.args)}
                            bindings.update({kw.arg:kw.value for kw in call.keywords if kw.arg})
                            for arg,value in bindings.items():
                                referenced=self.module_reference(caller,value)
                                if referenced:aliases.setdefault(arg,[]).append(referenced)
                aliases={k:v[0] for k,v in aliases.items() if len({x[0] for x in v})==1}
                for n in ast.walk(init):
                    if isinstance(n,ast.Assign):
                        value=n.value
                        if isinstance(value,ast.IfExp):
                            choice=constant(value.test,self.class_envs[c])
                            value=value.body if choice is not UNKNOWN and choice else value.orelse if choice is not UNKNOWN else value
                        reference=self.module_reference(c,value,aliases)
                        if isinstance(value,ast.Call) and dotted(value.func).split('.')[-1] in AFFINE|LOOKUP|NORM|ACT|{'Dropout','Identity','Sequential','ModuleList','ModuleDict'}|self.classes:
                            ctor=copy.deepcopy(value)
                            ctor.args=[copy.deepcopy(aliases[arg.id][1]) if isinstance(arg,ast.Name) and arg.id in aliases else arg for arg in ctor.args]
                            reference=(dotted(ctor.func).split('.')[-1],ctor)
                        if reference:
                            for target in n.targets:
                                if isinstance(target,ast.Name):aliases[target.id]=reference
                                elif isinstance(target,ast.Attribute) and dotted(target).startswith('self.'):
                                    if dotted(target).count('.')==1:
                                        # Literal declarations were collected
                                        # before instance mutations. Replaying
                                        # them here would undo a parent's later
                                        # nested replacement in class-order
                                        # dependent ways.
                                        indirect=not isinstance(n.value,ast.Call) or any(isinstance(arg,ast.Name) and arg.id in aliases for arg in n.value.args)
                                        if dotted(target) not in self.modules[c] or indirect:self.modules[c][dotted(target)]=reference
                                    else:
                                        owner=self.module_reference(c,target.value,aliases)
                                        if owner and owner[0] in self.methods:
                                            instances={(caller,getattr(ctor,'lineno',0),ast.dump(ctor)) for caller,modules in self.modules.items() for attr,(typ,ctor) in modules.items() if typ==owner[0] and attr.count('.')==1}
                                            if len(instances)>1:raise Unsupported('instance-specific nested replacement '+owner[0])
                                            self.modules[owner[0]]['self.'+target.attr]=reference
                        else:
                            for target in n.targets:
                                if isinstance(target,ast.Attribute) and dotted(target).startswith('self.') and dotted(target).count('.')>1:
                                    owner=self.module_reference(c,target.value,aliases)
                                    if owner and 'self.'+target.attr in self.modules.get(owner[0],{}):raise Unsupported('unresolved nested module replacement '+dotted(target))
                    if isinstance(n,ast.Assign) and dotted(n.value) in self.modules[c]:
                        for t in n.targets:
                            if dotted(t).startswith('self.'):self.modules[c][dotted(t)]=self.modules[c][dotted(n.value)]
                    if isinstance(n,ast.Assign) and isinstance(n.value,ast.Name) and n.value.id in aliases:
                        for t in n.targets:
                            if dotted(t).startswith('self.'):self.modules[c][dotted(t)]=aliases[n.value.id]
                    if isinstance(n,ast.Call) and dotted(n.func)=='object.__setattr__' and len(n.args)==3 and isinstance(n.args[1],ast.Constant) and isinstance(n.args[2],ast.Name) and n.args[2].id in aliases:
                        self.modules[c]['self.'+n.args[1].value]=aliases[n.args[2].id]
                    if isinstance(n,ast.Call) and dotted(n.func)=='object.__setattr__' and len(n.args)==3 and isinstance(n.args[1],ast.Constant):
                        owner=self.module_reference(c,n.args[0]);value=self.module_reference(c,n.args[2])
                        if owner and value and owner[0] in self.modules:self.modules[owner[0]]['self.'+n.args[1].value]=value
                local_modules={}
                for n in ast.walk(init):
                    if isinstance(n,ast.For) and isinstance(n.target,ast.Name):
                        sequence=self.module_reference(c,n.iter)
                        if sequence and sequence[0] in {'ModuleList','Sequential'}:
                            entries=sequence[1].args
                            if len(entries)==1 and isinstance(entries[0],ast.ListComp):entries=[entries[0].elt]
                            elif len(entries)==1 and isinstance(entries[0],(ast.List,ast.Tuple)):entries=entries[0].elts
                            if entries and all(isinstance(v,ast.Call) for v in entries) and len({dotted(v.func) for v in entries})==1:local_modules[n.target.id]=(dotted(entries[0].func).split('.')[-1],entries[0])
                for n in ast.walk(init):
                    if not isinstance(n,ast.Call) or not isinstance(n.func,ast.Attribute):continue
                    owner=self.module_reference(c,n.func.value,local_modules)
                    if not owner or owner[0] not in self.methods:continue
                    setter=self.methods[owner[0]].get(n.func.attr)
                    if not setter:continue
                    body=[s for s in setter.body if not isinstance(s,ast.Expr) or not isinstance(s.value,ast.Constant)]
                    if len(body)!=1 or not isinstance(body[0],ast.Expr) or not isinstance(body[0].value,ast.Call):continue
                    write=body[0].value
                    if dotted(write.func)!='object.__setattr__' or len(write.args)!=3 or dotted(write.args[0])!='self' or not isinstance(write.args[1],ast.Constant) or not isinstance(write.args[2],ast.Name):continue
                    bindings={a.arg:v for a,v in zip(setter.args.args[1:],n.args)}
                    bindings.update({k.arg:k.value for k in n.keywords if k.arg})
                    incoming=bindings.get(write.args[2].id)
                    reference=self.module_reference(c,incoming,local_modules) if incoming else None
                    if reference:
                        self.modules[owner[0]]['self.'+str(write.args[1].value)]=reference
                        self.initialization_aliases.append(ast.unparse(n))

    def module_reference(self,c,n,local_modules=None):
        """Resolve instantiated module aliases without executing constructors."""
        if isinstance(n,ast.Attribute) and not (isinstance(n.value,ast.Name) and n.value.id=='self'):
            owner=self.module_reference(c,n.value,local_modules)
            if owner:return self.modules.get(owner[0],{}).get('self.'+n.attr)
        if isinstance(n,ast.Subscript):
            sequence=self.module_reference(c,n.value,local_modules)
            index=constant(n.slice,self.class_envs[c])
            if sequence and sequence[0] in {'Sequential','ModuleList'} and isinstance(index,int):
                entries=sequence[1].args
                if len(entries)==1 and isinstance(entries[0],ast.ListComp):
                    comp=entries[0]
                    bound={v.id for g in comp.generators for v in ast.walk(g.target) if isinstance(v,ast.Name)}
                    if len(comp.generators)==1 and not comp.generators[0].ifs and isinstance(comp.elt,ast.Call) and not any(isinstance(v,ast.Name) and v.id in bound for v in ast.walk(comp.elt)):
                        return dotted(comp.elt.func).split('.')[-1],comp.elt
                    return None
                if len(entries)==1 and isinstance(entries[0],(ast.List,ast.Tuple)):entries=entries[0].elts
                if -len(entries)<=index<len(entries) and isinstance(entries[index],ast.Call):return dotted(entries[index].func).split('.')[-1],entries[index]
            return None
        if isinstance(n,ast.Name) and local_modules and n.id in local_modules:return local_modules[n.id]
        parts=dotted(n).split('.')
        if not parts:return None
        if parts[0]=='self':current=c;result=None
        elif local_modules and parts[0] in local_modules:result=local_modules[parts[0]];current=result[0]
        else:return None
        for attr in parts[1:]:
            result=self.modules.get(current,{}).get('self.'+attr)
            if not result:return None
            current=result[0]
        return result

    def inherited_method(self,c,name):
        if name in self.methods.get(c,{}):return True
        return any(self.inherited_method(dotted(base).split('.')[-1],name) for base in self.defs[c].bases if dotted(base).split('.')[-1] in self.methods)

    def metadata_attribute(self,c,name,seen=None):
        seen=set() if seen is None else set(seen)
        if name in seen:return False
        seen.add(name)
        value=self.attrs[c].get(name)
        if value is None:return name.split('.')[-1] in {'in_features','out_features','embedding_dim','num_embeddings','normalized_shape','eps','p','training'}
        init=self.methods[c].get('__init__')
        formals={a.arg for a in init.args.args[1:] if a.annotation and (dotted(a.annotation) in {'int','float','bool'} or dotted(a.annotation).endswith('Config'))} if init else set()
        def scalar(n):
            if isinstance(n,ast.Constant):return isinstance(n.value,(int,float,bool,type(None)))
            if isinstance(n,ast.Name):return n.id in formals
            if isinstance(n,ast.Attribute):
                if dotted(n).startswith('self.'):return self.metadata_attribute(c,dotted(n),seen)
                return isinstance(n.value,ast.Name) and n.value.id in formals
            if isinstance(n,ast.BinOp):return scalar(n.left) and scalar(n.right)
            if isinstance(n,ast.UnaryOp):return scalar(n.operand)
            if isinstance(n,ast.Call):return dotted(n.func) in {'int','float','bool','max','min'} and all(scalar(a) for a in n.args)
            return False
        return scalar(value)

    def record(self,c,m,n,kind):
        self.evidence.append({'file':self.files[c],'line':getattr(n,'lineno',1),'scope':c+'.'+m,'kind':kind,'code':ast.unparse(n).splitlines()[0][:220]})

    def expr(self,n,c,loc):
        if n is None:return P
        if isinstance(n,ast.Constant):return Val(('literal',n.value),kind='metadata')
        if isinstance(n,ast.Name):return loc.get(n.id, META)
        if isinstance(n,ast.Attribute):
            name=dotted(n)
            if n.attr not in {'shape','device','dtype','ndim','requires_grad'}:
                owner=self.module_reference(c,n.value)
                if owner:
                    typ,ctor=owner
                    if typ in self.methods:
                        method=self.methods[typ].get(n.attr)
                        if method and any(dotted(d)=='property' for d in method.decorator_list):return self.invoke_custom(typ,ctor,[],{},c,method_name=n.attr)
                        if 'self.'+n.attr in self.buffers[typ]:return FIXED
                        if self.metadata_attribute(typ,'self.'+n.attr):return META
                        if 'self.'+n.attr in self.attrs[typ] and 'self.'+n.attr not in self.modules[typ]:return P
                    if n.attr in {'weight','bias'} and typ in AFFINE|LOOKUP|NORM:return P
            if name.startswith('self.'):
                if name.count('.')>1 and n.attr in {'shape','device','dtype','ndim','requires_grad'}:return META
                if name in self.modules[c]:return Val(('module',c,name),kind='module')
                if name.count('.')==1 and n.attr in self.methods[c] and any(dotted(d)=='property' for d in self.methods[c][n.attr].decorator_list):
                    return self.method(c,n.attr,[])
                if name.count('.')==2 and '.'.join(name.split('.')[:2]) in self.modules[c]:
                    typ,_=self.modules[c]['.'.join(name.split('.')[:2])]
                    if typ in self.methods and n.attr in self.methods[typ] and any(dotted(d)=='property' for d in self.methods[typ][n.attr].decorator_list):return self.method(typ,n.attr,[])
                if name in self.class_envs.get(c,{}):return META
                if name in self.buffers[c]:return FIXED
                if self.metadata_attribute(c,name):return META
                return P
            base=self.expr(n.value,c,loc)
            if n.attr in {'shape','device','dtype','ndim','requires_grad'}:return META
            if base.kind=='module':raise Unsupported('unresolved indexed module attribute '+ast.unparse(n))
            if n.attr in {'T','mT'}:return node('transpose',base) if base.signal else P
            return node('attribute:'+n.attr,base) if base.signal else base
        if isinstance(n,(ast.Tuple,ast.List)):
            vals=tuple(self.expr(x,c,loc) for x in n.elts)
            return Val(('tuple',)+tuple(v.key for v in vals),any(v.signal for v in vals),'sequence')
        if isinstance(n,ast.Dict):
            vals=[self.expr(x,c,loc) for x in n.values]
            return node('dict',*vals)
        if isinstance(n,ast.DictComp):
            inner=dict(loc)
            for gen in n.generators:
                if self.expr(gen.iter,c,inner).signal:raise Unsupported('input-controlled dictionary comprehension')
                self.assign(gen.target,META,inner)
            values=[self.expr(n.key,c,inner),self.expr(n.value,c,inner)]
            if any(v.signal for v in values):raise Unsupported('signal dictionary comprehension')
            return Val(('dictionary',),kind='dictionary')
        if isinstance(n,ast.Subscript):
            v=self.expr(n.value,c,loc)
            if v.kind=='module':return v
            if v.kind=='metadata':return META
            if v.kind=='sequence' and isinstance(n.slice,ast.Constant) and type(n.slice.value)==int:
                return Val(v.key[n.slice.value+1],v.signal,'signal' if v.signal else 'parameter')
            if not v.signal:
                idx=self.expr(n.slice,c,loc)
                if idx.signal:
                    if dotted(n.value) in self.buffers[c]:self.buffer_lookups.add((c,dotted(n.value)))
                    self.signal_ops.add('lookup')
                    return node('lookup',idx)
                return v
            index=ast.dump(n.slice,include_attributes=False)
            if c not in self.roots and final_feature_slice(index) and expression_tags(v.key)&{'lookup','affine','normalize'}:
                self.record(c,self.stack[-1][1],n,'fixed final-feature coordinate selection within a traced hidden representation')
                return node('affine',affine_input(v))
            return node('index',v,index)
        if isinstance(n,ast.Slice):
            vals=[self.expr(x,c,loc) for x in [n.lower,n.upper,n.step]]
            return node('slice',*vals)
        if isinstance(n,ast.Starred):return self.expr(n.value,c,loc)
        if isinstance(n,ast.NamedExpr):
            value=self.expr(n.value,c,loc);self.assign(n.target,value,loc)
            return value
        if isinstance(n,ast.UnaryOp):
            v=self.expr(n.operand,c,loc)
            return node(type(n.op).__name__,v) if v.signal else P if learned(v) else v
        if isinstance(n,ast.BinOp):
            a,b=self.expr(n.left,c,loc),self.expr(n.right,c,loc)
            if not (a.signal or b.signal):
                self.parameter_ops.add(type(n.op).__name__)
                if isinstance(n.op,(ast.Mult,ast.MatMult,ast.Div,ast.Pow)) and learned(a) and learned(b):self.parameter_ops.add('learned-'+type(n.op).__name__)
                if isinstance(n.op,(ast.MatMult,ast.Mult)) and learned(a) and learned(b):self.parameter_product_sites.append({'scope':c+'.'+self.stack[-1][1],'operator':type(n.op).__name__,'roles':tuple(self.role_stack),'code':ast.unparse(n),'expanded_code':self.syntax(n)})
                return P if learned(a) or learned(b) else FIXED
            op=type(n.op).__name__
            self.signal_ops.add(op)
            if isinstance(n.op,(ast.Add,ast.Sub)) and a.signal!=b.signal:
                v=a if a.signal else b
                independent=n.right if a.signal else n.left
                expanded=self.syntax(independent,False)
                if c in self.roots and any(isinstance(item,ast.Call) and dotted(item.func).endswith('arange') for item in ast.walk(expanded)):
                    self.position_forms.append(ast.dump(NumericSettings().visit(expanded),include_attributes=False))
                if c in self.roots:
                    for item in ast.walk(expanded):
                        if isinstance(item,ast.Subscript) and dotted(item.value) in self.buffers[c]:
                            self.additive_buffers.add((c,dotted(item.value)))
                            self.position_forms.append('indexed additive feature buffer '+c+'.'+dotted(item.value))
                if v.key[0] in {'MatMult','matmul','einsum','bmm','scale','offset','partition'} and expression_tags(v.key)&{'MatMult','matmul','einsum','bmm'} and 'softmax' in ast.unparse(self.methods[c].get('forward',ast.Pass())):
                    self.score_offsets.append(ast.dump(NumericSettings().visit(expanded),include_attributes=False))
                    self.score_offset_geometry.append(self.geometry(independent,c))
                # Independent offsets alter affine coordinates; additive
                # positional paths are separately retained by model roles.
                affine_partition=v.key[0]=='partition' and v.key[1][0]=='affine'
                return v if v.key[0] in {'affine','normalize'} or affine_partition else node('offset',v)
            if isinstance(n.op,ast.Mult) and a.signal!=b.signal or isinstance(n.op,ast.Div) and a.signal and not b.signal:
                v=a if a.signal else b
                return v if v.key[0]=='normalize' else node('scale',v)
            if isinstance(n.op,ast.MatMult) and a.signal!=b.signal:
                origin=lookup_origin((a if a.signal else b).key)
                if origin is not None:return Val(('lookup',origin),True,'signal')
            return node(op,a,b)
        if isinstance(n,(ast.Compare,ast.BoolOp)):
            vals=[self.expr(x,c,loc) for x in ([n.left]+n.comparators if isinstance(n,ast.Compare) else n.values)]
            return node('comparison:'+ast.dump(n,include_attributes=False),*vals) if any(v.signal for v in vals) else META
        if isinstance(n,ast.IfExp):
            cv=constant(n.test,self.class_envs[c])
            if cv is not UNKNOWN:return self.expr(n.body if cv else n.orelse,c,loc)
            test=self.expr(n.test,c,loc)
            if test.signal:self.effects.add('input-dependent branch')
            a,b=self.expr(n.body,c,loc),self.expr(n.orelse,c,loc)
            return a if a==b else node('conditional',test,a,b)
        if isinstance(n,ast.Call):return self.call(n,c,loc)
        if isinstance(n,(ast.ListComp,ast.GeneratorExp)):
            inner=dict(loc)
            for gen in n.generators:self.assign(gen.target,META,inner)
            return self.expr(n.elt,c,inner)
        raise Unsupported('expression '+type(n).__name__)

    def invoke_module(self,typ,ctor,arg,c,loc):
        self.signal_ops.add(typ)
        if typ in AFFINE:
            arg=affine_input(arg)
            origin=lookup_origin(arg.key) if any(re.search(r'token_emb|token_embedding|token_projection|operand_digit_embedding|generated_digit_embedding',r) for r in self.role_stack) else None
            return Val(('lookup',origin),True,'signal') if origin is not None else node('affine',arg)
        if typ in LOOKUP:return node('lookup',arg) if arg.signal else P
        if typ in NORM:return node('normalize',arg)
        if typ in ACT:return node('activation:'+typ.lower(),arg)
        if typ in {'Dropout','Identity'}:return arg
        if typ in {'Sequential','ModuleList'}:
            modules=ctor.args
            if len(modules)==1 and isinstance(modules[0],(ast.List,ast.Tuple)):modules=modules[0].elts
            if len(modules)==1 and isinstance(modules[0],ast.ListComp):modules=[modules[0].elt]
            out=arg
            for inner in modules:
                if not isinstance(inner,ast.Call):raise Unsupported('nonliteral sequential module')
                out=self.invoke_module(dotted(inner.func).split('.')[-1],inner,out,c,loc)
            return out
        if typ in self.methods:return self.invoke_custom(typ,ctor,[arg],{},c)
        raise Unsupported('module '+typ)

    def invoke_custom(self,typ,ctor,args,kwargs,caller,method_name='forward'):
        """Bind constructor settings for this instance, then trace its call."""
        init=self.methods[typ].get('__init__')
        if init is None:return self.method(typ,method_name,args,kwargs)
        if any(isinstance(v,ast.Starred) for v in ctor.args) or any(k.arg is None for k in ctor.keywords):raise Unsupported('dynamic custom constructor expansion')
        previous=self.class_envs[typ];context=dict(previous)
        formals=init.args.args[1:]
        for formal in formals:context.pop(formal.arg,None)
        for name in self.attrs[typ]:context.pop(name,None)
        supplied={a.arg:v for a,v in zip(formals,ctor.args)}
        supplied.update({k.arg:k.value for k in ctor.keywords})
        defaults=dict(zip([a.arg for a in formals[-len(init.args.defaults):]],init.args.defaults)) if init.args.defaults else {}
        for formal in formals:
            expression=supplied.get(formal.arg,defaults.get(formal.arg))
            value=constant(expression,self.class_envs[caller]) if expression is not None else UNKNOWN
            if value is not UNKNOWN:context[formal.arg]=value
        for _ in range(3):
            for name,expression in self.attrs[typ].items():
                value=constant(expression,context)
                if value is not UNKNOWN:context[name]=value
                elif isinstance(expression,ast.IfExp):
                    choice=constant(expression.test,context)
                    selected=expression.body if choice is not UNKNOWN and choice else expression.orelse if choice is not UNKNOWN else None
                    if isinstance(selected,ast.Call):context[name]='instantiated module'
        self.class_envs[typ]=context
        try:return self.method(typ,method_name,args,kwargs)
        finally:self.class_envs[typ]=previous

    def call(self,n,c,loc):
        name=dotted(n.func);short=name.split('.')[-1]
        args=[self.expr(a,c,loc) for a in n.args]
        kwargs={k.arg:self.expr(k.value,c,loc) for k in n.keywords}
        if None in kwargs:raise Unsupported('dynamic keyword expansion '+name)
        if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Call) and dotted(n.func.value.func)=='super':
            bases=[dotted(b).split('.')[-1] for b in self.defs[c].bases]
            if len(bases)!=1:raise Unsupported('multiple-inheritance super call')
            if bases[0] in self.methods:return self.method(bases[0],short,args,kwargs)
            if short=='forward':return self.invoke_module(bases[0],n,args[0] if args else kwargs.get('input',P),c,loc)
            raise Unsupported('super method '+short)
        if name=='object.__getattribute__' and len(n.args)==2 and isinstance(n.args[1],ast.Constant):
            attr='self.'+str(n.args[1].value)
            if attr in self.modules[c]:return Val(('module',c,attr),kind='module')
            raise Unsupported('unresolved explicit attribute '+attr)
        if name.startswith('self.') and name.count('.')==1 and isinstance(self.attrs[c].get(name),ast.IfExp):
            expression=self.attrs[c][name];choice=constant(expression.test,self.class_envs[c])
            selected=expression.body if choice is not UNKNOWN and choice else expression.orelse if choice is not UNKNOWN else None
            if isinstance(selected,ast.Call):
                typ=dotted(selected.func).split('.')[-1]
                self.record(c,self.stack[-1][1],n,'constructor conditional resolved from this instance settings')
                if typ in self.methods:return self.invoke_custom(typ,selected,args,kwargs,c)
                if typ in AFFINE|LOOKUP|NORM|ACT|{'Dropout','Identity','Sequential'}:return self.invoke_module(typ,selected,args[0] if args else P,c,loc)
            raise Unsupported('unresolved constructor conditional '+name)
        if name.startswith('self.') and name in self.modules[c]:
            typ,ctor=self.modules[c][name]
            self.signal_ops.add(typ)
            incoming=args[0] if args else P
            self.role_stack.append(c+'.'+name)
            try:out=self.invoke_custom(typ,ctor,args,kwargs,c) if typ in self.methods else self.invoke_module(typ,ctor,incoming,c,loc)
            finally:self.role_stack.pop()
            def relative(k):
                if k in {incoming.key,affine_input(incoming).key}:return ('role input',)
                return tuple(relative(x) for x in k) if isinstance(k,tuple) else k
            self.roles[c+'.'+name]={'type':typ,'expression':relative(out.key),'input_expression':incoming.key,'constructor':ast.unparse(ctor)}
            return out
        if isinstance(n.func,ast.Subscript):
            mod=self.expr(n.func,c,loc)
            if mod.kind=='module':
                typ,ctor=self.modules[c][mod.key[2]]
                index=constant(n.func.slice,self.class_envs[c])
                entries=ctor.args[0].elts if ctor.args and isinstance(ctor.args[0],(ast.List,ast.Tuple)) else ctor.args
                if isinstance(index,int) and index<len(entries):
                    inner=entries[index];typ=dotted(inner.func).split('.')[-1]
                    return self.invoke_custom(typ,inner,args,kwargs,c) if typ in self.methods else self.invoke_module(typ,inner,args[0],c,loc)
        if name.startswith('self.') and name.count('.')==1 and self.inherited_method(c,short):return self.method(c,short,args,kwargs)
        if name.startswith('self.') and name.count('.')==2:
            owner='.'.join(name.split('.')[:2])
            if owner in self.modules[c]:
                typ,ctor=self.modules[c][owner]
                if typ in self.methods and short in self.methods[typ]:return self.method(typ,short,args,kwargs)
        if name.startswith('self.') and name.count('.')>=2:
            nested=self.module_reference(c,n.func)
            if nested:
                typ,ctor=nested;incoming=args[0] if args else kwargs.get('input',P)
                self.signal_ops.add(typ);self.role_stack.append(c+'.'+name)
                try:out=self.invoke_custom(typ,ctor,args,kwargs,c) if typ in self.methods else self.invoke_module(typ,ctor,incoming,c,loc)
                finally:self.role_stack.pop()
                def relative(k):
                    if k in {incoming.key,affine_input(incoming).key}:return ('role input',)
                    return tuple(relative(x) for x in k) if isinstance(k,tuple) else k
                self.roles[c+'.'+name]={'type':typ,'expression':relative(out.key),'input_expression':incoming.key,'constructor':ast.unparse(ctor)}
                return out
            referenced=self.module_reference(c,n.func.value)
            if referenced and referenced[0] in self.methods and short in self.methods[referenced[0]]:return self.method(referenced[0],short,args,kwargs)
            owner_value=self.expr(n.func.value,c,loc)
            if owner_value.kind=='module':
                typ,ctor=self.modules[owner_value.key[1]][owner_value.key[2]]
                if typ in self.methods and short in self.methods[typ]:return self.method(typ,short,args,kwargs)
        if isinstance(n.func,ast.Name) and n.func.id in loc and loc[n.func.id].kind=='module':
            mod=loc[n.func.id];typ,ctor=self.modules[mod.key[1]][mod.key[2]]
            incoming=args[0] if args else P
            out=self.invoke_custom(typ,ctor,args,kwargs,c) if typ in self.methods else self.invoke_module(typ,ctor,incoming,c,loc)
            def relative(k):
                if k==incoming.key:return ('role input',)
                return tuple(relative(v) for v in k) if isinstance(k,tuple) else k
            self.roles[c+'.'+mod.key[2]]={'type':typ,'expression':relative(out.key),'input_expression':incoming.key,'constructor':ast.unparse(ctor)}
            return out
        receiver=self.expr(n.func.value,c,loc) if isinstance(n.func,ast.Attribute) else None
        if (short in AFFINE|LOOKUP|NORM and name.startswith(('F.','torch.'))) or name in self.functional_aliases and self.functional_aliases[name] in AFFINE|LOOKUP|NORM:
            if any(v.signal for v in args[1:]) or any(v.signal for k,v in kwargs.items() if k not in {'input'}):raise Unsupported('input-dependent primitive coefficients '+name)
            incoming=args[0] if args else kwargs.get('input',P)
            primitive=self.functional_aliases.get(name,short)
            out=self.invoke_module(primitive,n,incoming,c,loc)
            if c in self.roots and primitive in LOOKUP and incoming.signal:
                def relative(k):
                    if k==INP.key:return ('role input',)
                    return tuple(relative(v) for v in k) if isinstance(k,tuple) else k
                owners=set()
                coefficient_scopes=set()
                coefficient=self.syntax(n.args[1],False) if len(n.args)>1 else None
                for item in ast.walk(coefficient) if coefficient is not None else []:
                    reference=self.module_reference(c,item)
                    if reference and reference[0] in self.methods:owners.add(reference[0])
                    if isinstance(item,ast.Call) and isinstance(item.func,ast.Attribute):
                        owner=c if dotted(item.func).count('.')==1 else (self.module_reference(c,item.func.value) or (None,None))[0]
                        if owner in self.methods and item.func.attr in self.methods[owner]:coefficient_scopes.add(owner+'.'+item.func.attr)
                typ=next(iter(owners)) if len(owners)==1 else 'Embedding'
                coefficient_calls={ast.dump(NumericSettings().visit(copy.deepcopy(item)),include_attributes=False) for item in ast.walk(coefficient) if isinstance(item,ast.Call)} if coefficient is not None else set()
                nonlinear_codes=[v['code'] for v in self.parameter_nonlinear_sites if v['scope']==c+'.'+self.stack[-1][1] and v['code'] in coefficient_calls]
                self.roles[c+'.functional_token_embedding:'+str(getattr(n,'lineno',0))]={'type':typ,'expression':relative(out.key),'input_expression':incoming.key,'constructor':ast.unparse(n),'coefficient_scopes':sorted(coefficient_scopes),'coefficient_nonlinear_codes':nonlinear_codes}
            return out
        if short in {'size','dim','numel','len','int','float','bool','isinstance','hasattr','getattr','arange'}:
            if short in {'int','float','bool'} and ((receiver and receiver.signal) or any(v.signal for v in args)):
                return node('cast:'+short,*([receiver] if receiver and receiver.signal else []),*args)
            if short in {'int','float','bool','getattr','arange'} and any(v.signal for v in args):raise Unsupported('data-dependent metadata operation '+name)
            return META
        if short in {'new_zeros','new_ones','new_empty','new_full','new_tensor','zeros_like','ones_like','empty_like','full_like','zeros','ones','empty','full','tensor','as_tensor','eye'}:
            if short in {'full','tensor','as_tensor'} and any(v.signal for v in args):raise Unsupported('data-dependent tensor construction '+name)
            if short=='full_like' and any(v.signal for v in args[1:]):raise Unsupported('data-dependent tensor fill '+name)
            if short in {'new_full','new_tensor'} and any(v.signal for v in args):raise Unsupported('data-dependent tensor construction '+name)
            return FIXED
        values=([receiver] if receiver else [])+args+[v for v in kwargs.values() if v.signal]
        signal=[v for v in values if v.signal]
        if short=='masked_fill' and 'softmax' in ast.unparse(self.methods[c].get('forward',ast.Pass())):
            self.score_forms.append(('score mask',self.syntax(n.args[0]) if n.args else 'missing'))
            self.geometry_forms.append(('score mask',self.geometry(n.args[0],c,mask=True) if n.args else 'missing'))
        if not signal and short in {'softmax','log_softmax'}:
            self.score_forms.append(('position-only score computation',self.syntax(n)))
            self.geometry_forms.append(('position-only score computation',self.geometry(n,c)))
            expanded=self.syntax(n,False)
            constructors={name for name,value in self.attrs[c].items() if isinstance(value,ast.Call) and dotted(value.func).endswith('.Parameter')}
            for call in ast.walk(expanded):
                if not isinstance(call,ast.Call) or dotted(call.func).split('.')[-1]!='irfft' or not call.args:continue
                core=call.args[0]
                if not any(isinstance(x,ast.Call) and dotted(x.func).split('.')[-1]=='rfft' for x in ast.walk(core)):continue
                for phase in ast.walk(core):
                    if not isinstance(phase,ast.Call) or dotted(phase.func).split('.')[-1]!='exp':continue
                    imaginary=any(isinstance(x,ast.Constant) and isinstance(x.value,complex) and x.value.imag for x in ast.walk(phase))
                    learned_shift=any(isinstance(x,ast.Attribute) and dotted(x) in constructors for x in ast.walk(phase))
                    frequency=any(isinstance(x,ast.Call) and dotted(x.func).split('.')[-1]=='arange' for x in ast.walk(phase))
                    if imaginary and learned_shift and frequency:
                        self.kernel_mechanisms.add('shared learned relative-distance base with learned spectral phase shifts')
                        self.record(c,'forward',n,'PDF algebraic-construction category: inverse-FFT kernels depend on nonlinear learned complex phase shifts')
        if not signal:
            if short in ACT|{'sin','cos','exp','softplus','log','log1p','pow','square','sqrt','rsqrt'} and any(learned(v) for v in values):
                self.parameter_nonlinear_sites.append({'scope':c+'.'+self.stack[-1][1],'operator':short,'roles':tuple(self.role_stack),'code':self.syntax(n)})
            if short=='get' and receiver and receiver.kind=='dictionary':return P
            if short in PARAM_OPS|ACT|{'sin','cos','exp','softmax','softplus','log','log1p','pow','clamp_min_','clamp_max_'}:
                self.parameter_ops.add(short)
                if short in {'matmul','einsum','bmm','mm','addmm','dot','outer'} and sum(learned(v) for v in values)>1:self.parameter_ops.add('learned-MatMult')
                if short in {'mul','div'} and sum(learned(v) for v in values)>1:self.parameter_ops.add('learned-'+short.title())
                return P if any(learned(v) for v in values) else FIXED
            # A free helper is only safe to inspect when its implementation is
            # present. Merely having no signal argument is not a proof.
            if name in self.defs and isinstance(self.defs[name],ast.FunctionDef):
                return self.free_method(name,args,c)
            raise Unsupported('parameter callable '+name)
        self.signal_ops.add(short)
        if short in {'add','sub','mul','div','remainder'}:
            operands=args if name.startswith(('torch.','F.')) else ([receiver] if receiver else [])+args
            if len(operands)<2:raise Unsupported('incomplete tensor arithmetic '+name)
            a,b=operands[:2]
            floor=short=='div' and any(k.arg=='rounding_mode' and isinstance(k.value,ast.Constant) and k.value.value=='floor' for k in n.keywords)
            op={'add':'Add','sub':'Sub','mul':'Mult','div':'FloorDiv' if floor else 'Div','remainder':'Mod'}[short]
            if op=='Mult' and a.signal!=b.signal or op=='Div' and a.signal and not b.signal:return node('scale',a if a.signal else b)
            if op in {'Add','Sub'} and a.signal!=b.signal:
                v=a if a.signal else b
                return v if v.key[0]=='affine' else node('offset',v)
            return node(op,a,b)
        if short in {'lt','le','gt','ge','eq','ne','logical_and','logical_or','logical_not','long','int'}:
            return node(short,*values)
        if short in LAYOUT:
            return signal[0] # Storage layout; semantic axis operations below stay.
        if short=='expand' and len(n.args)==4 and isinstance(n.args[2],ast.Attribute) and dotted(n.args[2])=='self.n_head' and self.metadata_attribute(c,'self.n_head'):
            receiver_ast=n.func.value if isinstance(n.func,ast.Attribute) else None
            inserted_head=isinstance(receiver_ast,ast.Call) and isinstance(receiver_ast.func,ast.Attribute) and receiver_ast.func.attr=='unsqueeze' and len(receiver_ast.args)==1 and constant(receiver_ast.args[0],self.class_envs[c])==2
            kept_axes=all(constant(n.args[i],self.class_envs[c])==-1 for i in [0,1,3])
            if inserted_head and kept_axes and coordinate_signal_key(signal[0].key)[0]=='affine':
                self.record(c,self.stack[-1][1],n,'shared affine projection repeated over explicit attention heads; batch, token and feature axes unchanged')
                return signal[0]
        if short=='expand' and len(n.args)==2 and isinstance(n.args[0],ast.Starred) and isinstance(n.args[0].value,ast.Subscript):
            shape=n.args[0].value
            if isinstance(shape.value,ast.Attribute) and shape.value.attr=='shape' and isinstance(shape.slice,ast.Slice) and shape.slice.lower is None and shape.slice.step is None and constant(shape.slice.upper,{})==-1 and isinstance(n.func,ast.Attribute) and ast.dump(shape.value.value)==ast.dump(n.func.value):
                self.record(c,self.stack[-1][1],n,'repeat only final feature/logit coordinates; preceding axes unchanged')
                return signal[0]
        if short in {'chunk','split','unbind'}:
            base=signal[0]
            count=constant(n.args[0],self.class_envs[c]) if n.args else UNKNOWN
            count=count if isinstance(count,int) and 0<count<32 else 3
            vals=[base if base.key[0]=='affine' else node('partition',base,i) for i in range(count)]
            return Val(('tuple',)+tuple(v.key for v in vals),True,'sequence')
        if short in ACT:return node('activation:'+short.lower(),signal[0])
        if short=='glu':return node('gated-glu',signal[0])
        if short=='index_copy' and receiver and not receiver.signal and len(args)==3 and not args[1].signal and args[2].signal and constant(n.args[0],self.class_envs[c])==-1 and any(re.search(r'\.output$|lm_head|output_projection|classifier',r) for r in self.role_stack):
            self.output_forms.add('fixed vocabulary subset; affine valid-token logits and constant invalid-token logits')
            return node('affine',affine_input(args[2]))
        if short in {'dropout'}:return signal[0]
        if short in {'softmax','log_softmax'}:
            return node(short,signal[0],ast.dump(n.keywords[0].value,include_attributes=False) if n.keywords else ast.dump(n.args[-1],include_attributes=False) if args else -1)
        if short=='cat':
            value=node(short,*values,tuple((k.arg,ast.dump(k.value,include_attributes=False)) for k in n.keywords))
            def anchors(k):
                if not isinstance(k,tuple):return []
                own=[k[1]] if k and k[0]=='affine' else []
                return own+sum((anchors(x) for x in k),[])
            for anchor in anchors(value.key):
                if feature_affine(value.key,anchor):
                    if c in self.roots or any(re.search(r'\.output$|lm_head|classifier',r) for r in self.role_stack):self.output_forms.add('affine vocabulary readout with explicit fixed/shared logit coordinates')
                    return node('affine',Val(anchor,True,'signal'))
        if short=='pad' and len(n.args)>=2 and isinstance(n.args[1],(ast.Tuple,ast.List)) and len(n.args[1].elts)==2 and all(isinstance(x,ast.Constant) and isinstance(x.value,int) for x in n.args[1].elts) and args[0].key[0]=='lookup':
            return args[0] # Only the last feature axis of a table is extended.
        if short=='where' and len(args)==3:
            yes,no=[categorical_lookup_origin(v.key) for v in args[1:]]
            if yes is not None and yes==no:
                self.record(c,self.stack[-1][1],n,'piecewise storage of the same categorical token table; no digit decomposition')
                return Val(('lookup',yes),True,'signal')
        if short=='sum' and self.is_attention_head_sum(n,c):
            self.record(c,self.stack[-1][1],n,'sum over the explicitly constructed head axis, not token history')
            return node('attention-head-sum',signal[0])
        if short in {'transpose','permute','movedim','flip','roll','sum','mean','prod','amax','amin','max','min','cumsum','cumprod','repeat','repeat_interleave','expand','expand_as','gather','scatter','index_select','index_copy','select','narrow','masked_fill','where','clamp','clamp_min','clamp_max','minimum','maximum','sin','cos','sinc','exp','pow','square','sqrt','rsqrt','abs','cat','stack','pad','matmul','einsum','bmm','mm','addmm','tril','triu','one_hot'}:
            return node(short,*values,tuple((k.arg,ast.dump(k.value,include_attributes=False)) for k in n.keywords))
        if name in self.defs and isinstance(self.defs[name],ast.FunctionDef):return self.free_method(name,args,c)
        raise Unsupported('signal callable '+name)

    def assign(self,target,value,loc):
        if isinstance(target,ast.Name):loc[target.id]=value
        elif isinstance(target,(ast.Tuple,ast.List)):
            for i,t in enumerate(target.elts):
                item=Val(value.key[i+1],value.signal,'signal' if value.signal else 'parameter') if value.kind=='sequence' and len(value.key)>i+1 else META if value.kind=='metadata' else node('partition',value,i)
                self.assign(t,item,loc)
        elif isinstance(target,(ast.Subscript,ast.Attribute)):
            base=target.value
            if value.signal or isinstance(target,ast.Attribute) and dotted(target).startswith('self.') or any(isinstance(n,ast.Name) and n.id in loc and loc[n.id].signal for n in ast.walk(base)):
                raise Unsupported('signal/state mutation '+ast.unparse(target))

    def syntax(self,n,as_dump=True):
        aliases=self.syntax_stack[-1] if self.syntax_stack else {}
        class Expand(ast.NodeTransformer):
            def visit_Name(self,item):return copy.deepcopy(aliases.get(item.id,item))
        expanded=Expand().visit(copy.deepcopy(n))
        return ast.dump(NumericSettings().visit(expanded),include_attributes=False) if as_dump else expanded

    def is_attention_head_sum(self,n,c):
        """Prove the common [batch, heads, tokens, features] reduction locally.

        A sum on axis 1 is otherwise a possible history/state operation. Only
        an explicit four-axis value layout feeding attention establishes that
        axis as heads; names or the presence of softmax alone do not suffice.
        """
        dim=next((k.value for k in n.keywords if k.arg=='dim'),n.args[0] if n.args else None)
        if dim is None or constant(dim,self.class_envs[c])!=1 or not isinstance(n.func,ast.Attribute):return False
        method=self.methods[c].get(self.stack[-1][1])
        if method is None:return False
        bindings=[(t.id,s.value,s.lineno) for s in method.body if isinstance(s,(ast.Assign,ast.AnnAssign)) for t in (s.targets if isinstance(s,ast.Assign) else [s.target]) if isinstance(t,ast.Name)]
        def resolve(v,line):
            for _ in range(30):
                matches=[(value,at) for name,value,at in bindings if isinstance(v,ast.Name) and name==v.id and at<line]
                if not matches:return v,line
                v,line=max(matches,key=lambda pair:pair[1])
            return ast.Constant(None),line
        def value_heads(v,line):
            v,line=resolve(v,line)
            if not isinstance(v,ast.Call) or not isinstance(v.func,ast.Attribute) or v.func.attr!='transpose' or len(v.args)!=2:return False
            if [constant(a,self.class_envs[c]) for a in v.args]!=[1,2]:return False
            layout,_=resolve(v.func.value,line)
            if not isinstance(layout,ast.Call) or not isinstance(layout.func,ast.Attribute) or layout.func.attr not in {'expand','view','reshape'} or len(layout.args)!=4:return False
            return isinstance(layout.args[2],ast.Attribute) and dotted(layout.args[2])=='self.n_head' and self.metadata_attribute(c,'self.n_head')
        def has_softmax(v,line,depth=0):
            if depth>20:return False
            v,line=resolve(v,line)
            if isinstance(v,ast.Call) and dotted(v.func).split('.')[-1]=='softmax':return True
            return any(has_softmax(x,line,depth+1) for x in ast.iter_child_nodes(v))
        def att_value(v,line):
            v,line=resolve(v,line)
            if isinstance(v,ast.Call) and dotted(v.func)=='torch.einsum' and len(v.args)==3 and isinstance(v.args[0],ast.Constant) and v.args[0].value=='bhli,hoi->bhlo':
                mixed,at=resolve(v.args[1],line)
                if isinstance(mixed,ast.BinOp) and isinstance(mixed.op,ast.MatMult):
                    left,right=mixed.left,mixed.right
                    def unsqueezed(x,axis):return isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='unsqueeze' and len(x.args)==1 and constant(x.args[0],self.class_envs[c])==axis
                    # [heads, query, key] gains batch axis; the input sequence
                    # gains head axis. The equation retains that head axis.
                    if unsqueezed(left,0) and unsqueezed(right,1):return True
            if isinstance(v,ast.BinOp) and isinstance(v.op,ast.MatMult):
                return value_heads(v.right,line) and has_softmax(v.left,line)
            if isinstance(v,ast.BinOp) and isinstance(v.op,(ast.Mult,ast.Div)):
                # The other factor must contain no input-dependent expression;
                # the interpreter already records its signal graph separately.
                return att_value(v.left,line) or att_value(v.right,line)
            return False
        return att_value(n.func.value,n.lineno)

    def geometry(self,n,c,mask=False):
        """Trace positional index algebra while quotienting table coordinates.

        Coefficients can be centered, tied, padded or reconstructed without
        changing a distance lookup. Query/key direction, masks, nonlinear
        position functions and unrecognized helper bodies remain explicit.
        """
        root=self.syntax(n,False)
        coefficient=('coefficient',)
        def table_index(key):
            if not isinstance(key,tuple) or not key:return key
            values=tuple(table_index(v) for v in key)
            if values[0]=='clamp_max' and len(values)>=2:return values[1]
            if values[0]=='clamp' and len(values)>=3 and values[2]==('constant',0):return ('clamp_min',values[1],('constant',0))
            return values
        cached_lags={}
        init=self.methods[c].get('__init__')
        if init:
            assigned={t.id:s.value for s in init.body if isinstance(s,ast.Assign) for t in s.targets if isinstance(t,ast.Name)}
            class ExpandConstructor(ast.NodeTransformer):
                def __init__(self):self.seen=set()
                def visit_Name(self,item):
                    if item.id not in assigned or item.id in self.seen:return item
                    self.seen.add(item.id);out=self.visit(copy.deepcopy(assigned[item.id]));self.seen.remove(item.id);return out
            for attr in self.buffers[c]:
                definition=self.attrs[c].get(attr)
                if definition is None:continue
                expanded=ExpandConstructor().visit(copy.deepcopy(definition))
                # A cached causal distance grid is exactly the same pairwise
                # index construction; arbitrary learned/static buffers are not.
                core=expanded.func.value if isinstance(expanded,ast.Call) and isinstance(expanded.func,ast.Attribute) and expanded.func.attr=='clamp_min' and len(expanded.args)==1 and constant(expanded.args[0],{})==0 else None
                if not isinstance(core,ast.BinOp) or not isinstance(core.op,ast.Sub):continue
                if not all(isinstance(v,ast.Subscript) and isinstance(v.value,ast.Call) and dotted(v.value.func)=='torch.arange' for v in [core.left,core.right]):continue
                if ast.dump(core.left.value)!=ast.dump(core.right.value):continue
                expected=[ast.parse('positions[:,None]',mode='eval').body.slice,ast.parse('positions[None,:]',mode='eval').body.slice]
                if [ast.dump(core.left.slice),ast.dump(core.right.slice)]==[ast.dump(x) for x in expected]:cached_lags[attr]=expanded
        def inspect(item):
            if isinstance(item,ast.Constant):return ('constant',item.value),False
            if isinstance(item,ast.Name):return coefficient,False
            if isinstance(item,ast.Attribute):
                if dotted(item) in cached_lags:return inspect(cached_lags[dotted(item)])
                if dotted(item).startswith('self.'):return coefficient,False
                base,dependent=inspect(item.value)
                return (('attribute',item.attr,base),True) if dependent else (coefficient,False)
            if isinstance(item,(ast.Tuple,ast.List)):
                vals=[inspect(x) for x in item.elts]
                return ('tuple',)+tuple(k for k,d in vals),any(d for k,d in vals)
            if isinstance(item,ast.Subscript):
                base,bd=inspect(item.value);index,idp=inspect(item.slice)
                if dotted(item.value) in cached_lags and isinstance(item.slice,ast.Tuple) and len(item.slice.elts)==2:
                    a,b=item.slice.elts
                    if isinstance(a,ast.Slice) and isinstance(b,ast.Slice) and a.lower is b.lower is a.step is b.step is None and ast.dump(a)==ast.dump(b):return base,bd
                if not bd and idp:
                    if isinstance(item.slice,ast.Tuple):
                        indices=[inspect(v) for v in item.slice.elts]
                        live=[k for k,d in indices if d]
                        if len(live)==1 and all(d or k in {('slice',('absent',),('absent',),('absent',)),('constant',None)} for k,d in indices):index=live[0]
                    return ('positional table lookup',table_index(index)),True
                return (('index',base,index),True) if bd else (coefficient,False)
            if isinstance(item,ast.Slice):
                vals=[inspect(x) if x is not None else (('absent',),False) for x in (item.lower,item.upper,item.step)]
                return ('slice',)+tuple(k for k,d in vals),any(d for k,d in vals)
            if isinstance(item,(ast.BinOp,ast.UnaryOp,ast.Compare,ast.BoolOp)):
                children=([item.left,item.right] if isinstance(item,ast.BinOp) else [item.operand] if isinstance(item,ast.UnaryOp) else [item.left]+item.comparators if isinstance(item,ast.Compare) else item.values)
                vals=[inspect(x) for x in children]
                op=','.join(type(x).__name__ for x in item.ops) if isinstance(item,ast.Compare) else type(item.op).__name__
                return ((op,)+tuple(k for k,d in vals),True) if any(d for k,d in vals) else (coefficient,False)
            if isinstance(item,ast.Call):
                name=dotted(item.func);short=name.split('.')[-1]
                if short=='arange':
                    # FFT frequency coordinates derive from coefficient-table
                    # size, not the input token positions. Their mechanism is
                    # retained separately by the spectral-kernel witness.
                    if any(isinstance(x,ast.Call) and dotted(x.func).endswith('.numel') for arg in item.args for x in ast.walk(arg)):return coefficient,False
                    return ('position axis',),True
                args=[inspect(x) for x in item.args]
                recv=inspect(item.func.value) if isinstance(item.func,ast.Attribute) and not name.startswith(('torch.','F.','self.')) else (coefficient,False)
                vals=[recv]+args;dependent=any(d for k,d in vals)
                if short in {'embedding'} and args and args[0][1]:return ('positional table lookup',table_index(args[0][0])),True
                if name in self.modules[c] and self.modules[c][name][0] in LOOKUP and args and args[0][1]:return ('positional table lookup',table_index(args[0][0])),True
                if short=='cat' and args and isinstance(args[0][0],tuple) and args[0][0] and args[0][0][0]=='tuple' and any(k.arg=='dim' and constant(k.value,{})==-1 for k in item.keywords):
                    pieces=args[0][0][1:]
                    if pieces and all(p==pieces[0] and isinstance(p,tuple) and p and p[0]=='positional table lookup' for p in pieces):return pieces[0],True
                if short in LAYOUT and recv[1]:return recv
                if short=='permute' and len(item.args)==3 and [constant(v,{}) for v in item.args]==[2,0,1] and recv[1]:return recv
                if name.startswith('self.') and name not in self.modules[c] and short not in PARAM_OPS|ACT|{'exp','sin','cos','log','log1p','pow','softplus'}:
                    owner=c if name.count('.')==1 else (self.module_reference(c,item.func.value) or (None,None))[0]
                    method=self.methods.get(owner,{}).get(short)
                    return ('explicit coefficient helper',ast.dump(method,include_attributes=False) if method else ast.dump(item,include_attributes=False)),True
                if not dependent and short not in {'softmax','log_softmax','tril','triu'}:return coefficient,False
                return (short,)+tuple(k for k,d in vals)+tuple((kw.arg,ast.dump(kw.value,include_attributes=False)) for kw in item.keywords),True
            return ('unresolved positional syntax',ast.dump(item,include_attributes=False)),True
        key,dependent=inspect(root)
        return ('explicit fixed mask',ast.dump(root,include_attributes=False)) if mask and not dependent else key

    def body(self,statements,c,loc):
        for statement_index,n in enumerate(statements):
            if isinstance(n,(ast.Assign,ast.AnnAssign)):
                # Loss-only branches do not feed model predictions.
                targets=n.targets if isinstance(n,ast.Assign) else [n.target]
                if all(dotted(t) in {'loss'} for t in targets):continue
                value=self.expr(n.value,c,loc)
                expanded=self.syntax(n.value,False) if not value.signal else None
                for target in targets:
                    self.assign(target,value,loc)
                    if self.syntax_stack and isinstance(target,ast.Name):
                        if expanded is not None:self.syntax_stack[-1][target.id]=expanded
                        else:self.syntax_stack[-1].pop(target.id,None)
            elif isinstance(n,ast.AugAssign):
                v=self.expr(ast.BinOp(n.target,n.op,n.value),c,loc);self.assign(n.target,v,loc)
            elif isinstance(n,ast.Return):
                if isinstance(n.value,ast.Tuple) and len(n.value.elts)==2 and isinstance(n.value.elts[1],ast.Name) and n.value.elts[1].id=='loss':return self.expr(n.value.elts[0],c,loc)
                return self.expr(n.value,c,loc)
            elif isinstance(n,ast.If):
                if 'targets' in ast.unparse(n.test) and all(isinstance(v,(ast.Assign,ast.AnnAssign)) and all(dotted(t)=='loss' for t in v.targets if isinstance(v,ast.Assign)) for v in n.body):continue
                if all(isinstance(v,(ast.Raise,ast.Assert)) for v in n.body):continue
                test=self.expr(n.test,c,loc)
                if test.signal:self.effects.add('input-dependent branch')
                cv=constant(n.test,self.class_envs[c])
                if cv is not UNKNOWN:
                    returned=self.body(n.body if cv else n.orelse,c,loc)
                    if returned is not None:return returned
                    continue
                left,right=dict(loc),dict(loc)
                a,b=self.body(n.body,c,left),self.body(n.orelse,c,right)
                if a is not None or b is not None:
                    if a is None:a=self.body(statements[statement_index+1:],c,left)
                    if b is None:b=self.body(statements[statement_index+1:],c,right)
                    if a==b:return a
                    if not test.signal and a is not None and b is not None and a.signal and b.signal:
                        for anchor in loc.values():
                            if anchor.signal and feature_affine(a.key,anchor.key) and feature_affine(b.key,anchor.key):
                                optional=isinstance(n.test,ast.Compare) and all(isinstance(op,(ast.Is,ast.IsNot)) for op in n.test.ops) and all(isinstance(v,ast.Constant) and v.value is None for v in n.test.comparators)
                                self.parameter_ops.add('optional stored affine coordinates' if optional else 'conditional coefficient')
                                return node('affine',affine_input(anchor))
                    if not test.signal and a is not None and b is not None and not a.signal and not b.signal:
                        self.parameter_ops.add('conditional coefficient')
                        return P if learned(a) or learned(b) else FIXED
                    raise Unsupported('unresolved return branch '+ast.unparse(n.test))
                for key in left.keys()|right.keys():
                    l,r=left.get(key,P),right.get(key,P)
                    loc[key]=l if l==r else node('branch:'+ast.dump(n.test,include_attributes=False),test,l,r)
            elif isinstance(n,ast.For):
                it=self.expr(n.iter,c,loc)
                self.assign(n.target,it if it.kind=='module' else META,loc)
                if it.kind!='module' and any(isinstance(x,ast.Name) and x.id in loc and loc[x.id].signal for x in ast.walk(n.iter)):
                    raise Unsupported('input-controlled loop')
                returned=self.body(n.body,c,loc)
                if returned is not None:return returned
            elif isinstance(n,ast.Expr):
                if isinstance(n.value,ast.Constant):continue
                # Calls can mutate a signal even if their return is discarded.
                v=self.expr(n.value,c,loc)
                if v.signal:raise Unsupported('discarded signal call')
            elif isinstance(n,(ast.Pass,ast.Assert)):continue
            elif isinstance(n,ast.Delete):
                if any(not isinstance(t,ast.Name) for t in n.targets):raise Unsupported('nonlocal deletion')
                for t in n.targets:loc.pop(t.id,None)
            elif isinstance(n,ast.With):
                returned=self.body(n.body,c,loc)
                if returned is not None:return returned
            else:raise Unsupported('statement '+type(n).__name__)
        return None

    def method(self,c,name,args,kwargs=None):
        key=(c,name)
        if key in self.stack:raise Unsupported('recursive method')
        method=self.methods[c].get(name)
        if method is None:
            for base in self.defs[c].bases:
                typ=dotted(base).split('.')[-1]
                if typ in AFFINE|LOOKUP|NORM and name=='forward':return self.invoke_module(typ,ast.Call(ast.Name(typ),[],[]),args[0],c,{})
                if typ in self.methods:return self.method(typ,name,args,kwargs)
            raise Unsupported('inherited method '+c+'.'+name)
        self.stack.append(key)
        self.syntax_stack.append({})
        try:
            formals=method.args.args if any(dotted(d)=='staticmethod' for d in method.decorator_list) else method.args.args[1:]
            loc={a.arg:v for a,v in zip(formals,args)}
            if kwargs:loc.update(kwargs)
            defaults={a.arg:d for a,d in zip(formals[-len(method.args.defaults):],method.args.defaults)} if method.args.defaults else {}
            for a in formals:
                if a.arg not in loc:
                    if a.arg in defaults:loc[a.arg]=self.expr(defaults[a.arg],c,loc)
                    else:raise Unsupported('unbound method argument '+c+'.'+name+'.'+a.arg)
            out=self.body(method.body,c,loc)
            if out and out.signal and name=='forward' and args and any(re.search(r'\.output$|lm_head|output_projection|classifier',r) for r in self.role_stack):
                prototype_input=self.prototype_readout(method,c,loc)
                if prototype_input is not None:
                    self.output_forms.add('negative squared-Euclidean prototype logits; affine scores up to a common softmax offset')
                    out=node('affine',affine_input(prototype_input))
            if args and args[0].signal and out and out.signal and out.kind!='sequence' and feature_affine(out.key,args[0].key):out=node('affine',affine_input(args[0]))
            self.record(c,name,method,'symbolically followed runtime method')
            return out or P
        finally:self.stack.pop();self.syntax_stack.pop()

    def prototype_readout(self,method,c,loc):
        """Recognize the exact -sum((h[...,None,:]-p)**2) output identity.

        The common -||h||² term cancels across vocabulary logits; the remaining
        scores are affine in h, with biases tied to prototype norms.
        """
        assigned={t.id:n.value for n in method.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
        returns=[n.value for n in method.body if isinstance(n,ast.Return)]
        if len(returns)!=1:return None
        value=returns[0]
        if not isinstance(value,ast.UnaryOp) or not isinstance(value.op,ast.USub):return None
        summed=value.operand
        if not isinstance(summed,ast.Call) or not isinstance(summed.func,ast.Attribute) or summed.func.attr!='sum':return None
        dim=next((k.value for k in summed.keywords if k.arg=='dim'),summed.args[0] if summed.args else ast.Constant(None))
        if constant(dim,self.class_envs[c])!=-1:return None
        squared=summed.func.value
        if not isinstance(squared,ast.Call) or not isinstance(squared.func,ast.Attribute) or squared.func.attr!='square':return None
        diff=squared.func.value
        if isinstance(diff,ast.Name):diff=assigned.get(diff.id,diff)
        if not isinstance(diff,ast.BinOp) or not isinstance(diff.op,ast.Sub):return None
        left=diff.left
        if not isinstance(left,ast.Call) or not isinstance(left.func,ast.Attribute) or left.func.attr!='unsqueeze' or not left.args or constant(left.args[0],self.class_envs[c])!=-2:return None
        source=self.expr(left.func.value,c,loc);coefficient=self.expr(diff.right,c,loc)
        return source if source.signal and not coefficient.signal else None

    def free_method(self,name,args,c):
        if ('free',name) in self.stack:raise Unsupported('recursive helper')
        m=self.defs[name];self.stack.append(('free',name));self.syntax_stack.append({})
        try:return self.body(m.body,c,{a.arg:v for a,v in zip(m.args.args,args)}) or P
        finally:self.stack.pop();self.syntax_stack.pop()

    def inspect(self):
        root=next(iter(self.roots))
        result=self.method(root,'forward',[INP])
        enc={}
        enc_descriptors={}
        for name in ['encode_inputs','encode_targets','decode_targets']:
            if name in self.defs:
                # Exact encodings stay visible, including token constants.
                descriptor=encoding_descriptor(self.defs[name]) if name=='encode_inputs' else target_descriptor(self.defs[name],self.sources)
                if descriptor:
                    enc_descriptors[name]=descriptor
                    self.evidence.extend(descriptor['evidence'])
                enc[name]=(descriptor['family'] if descriptor else None) or formatting_family(name,self.defs[name]) or ast.dump(self.defs[name],include_attributes=False)
        construction=self.construction_features()
        return {'signal_signature':digest((coordinate_signal_key(result.key),enc,sorted(set(self.geometry_forms),key=repr))), 'signal_expression':result.key,
                'encodings':enc,'encoding_descriptors':enc_descriptors,'signal_operations':sorted(self.signal_ops),
                'parameter_operations':sorted(self.parameter_ops),'evidence':self.evidence,
                'construction_features':construction,'roles':self.roles,
                'effects':sorted(self.effects),'score_forms':sorted(set(self.score_forms)),
                'geometry_forms':sorted(set(self.geometry_forms),key=repr),
                'position_forms':sorted(set(self.position_forms)),
                'score_offsets':sorted(set(self.score_offsets)),
                'score_offset_geometry':sorted(set(self.score_offset_geometry),key=repr),
                'output_forms':sorted(self.output_forms),
                'initialization_aliases':sorted(set(self.initialization_aliases)),
                'kernel_mechanisms':sorted(self.kernel_mechanisms),
                'parameter_nonlinear_sites':self.parameter_nonlinear_sites,
                'parameter_product_sites':self.parameter_product_sites,
                'routing_geometry':self.routing_geometry,'root':root,'configuration_decisions':self.decisions}

    def construction_features(self):
        """Keep fixed generators introduced during construction visible.

        Their forward call can still look like a plain linear/lookup operator;
        omitting constructor provenance would erase the handoff's key example.
        """
        used={e['scope'].split('.')[0] for e in self.evidence}
        features=set()
        self.routing_geometry=[]
        for c in used:
            if c not in self.methods:continue
            init=self.methods[c].get('__init__')
            if not init:continue
            local={}
            mutations={}
            for n in ast.walk(init):
                if isinstance(n,ast.Assign):
                    for t in n.targets:
                        if isinstance(t,ast.Name):local[t.id]=n.value
                        if isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name):mutations.setdefault(t.value.id,[]).append(n.value)
            for n in ast.walk(init):
                if not isinstance(n,ast.Call):continue
                name=dotted(n.func)
                if name=='self.register_buffer' and len(n.args)>1:
                    pending=[n.args[1]];seen=set();calls=set();pieces=[];operators=set();radix_ops=set()
                    while pending:
                        item=pending.pop()
                        pieces.append(ast.dump(NumericSettings().visit(copy.deepcopy(item)),include_attributes=False))
                        for v in ast.walk(item):
                            if isinstance(v,ast.BinOp):
                                operators.add(type(v.op).__name__)
                                if isinstance(v.right,ast.Constant) and v.right.value==10:radix_ops.add(type(v.op).__name__)
                            if isinstance(v,ast.Call):
                                cal=dotted(v.func);calls.add(cal.split('.')[-1])
                                if cal in self.defs and cal not in seen:seen.add(cal);pending.append(self.defs[cal])
                            if isinstance(v,ast.Name) and v.id in local and v.id not in seen:
                                seen.add(v.id);pending.append(local[v.id]);pending.extend(mutations.get(v.id,[]))
                    generators=calls&{'sin','cos','exp','log','log1p','pow','vander','fft','rfft'}
                    if generators:
                        position_types={v['type'] for role,v in self.roles.items() if re.search(r'pos_emb|position_emb',role)}
                        token_types={v['type'] for role,v in self.roles.items() if re.search(r'token_emb|token_embedding',role)}
                        attr='self.'+str(n.args[0].value) if isinstance(n.args[0],ast.Constant) else ''
                        additive=(c,attr) in self.additive_buffers and (c,attr) not in self.buffer_lookups or c in position_types-token_types
                        features.add(('additive position feature generator' if additive else 'fixed algebraic generator',tuple(sorted(generators))))
                        self.record(c,'__init__',n,'fixed buffer construction follows '+','.join(sorted(generators)))
                    if isinstance(n.args[0],ast.Constant) and (c,'self.'+str(n.args[0].value)) in self.buffer_lookups and {'minimum','maximum'}<=calls and {'FloorDiv','Mod'}<=radix_ops:
                        features.add(('operand-swap canonical index map',))
                        self.record(c,'__init__',n,'input-indexed fixed buffer canonicalizes decimal digit pairs by minimum/maximum')
                    if isinstance(n.args[0],ast.Constant) and n.args[0].value=='weight':features.add(('fixed coefficient matrix',))
                    forward=self.methods[c].get('forward')
                    if isinstance(n.args[0],ast.Constant) and 'mask' in str(n.args[0].value).lower() and forward and 'softmax' in ast.unparse(forward):
                        self.routing_geometry.append({'operators':sorted(calls),'construction':sorted(pieces)})
                if name.endswith('.Parameter') and any(k.arg=='requires_grad' and isinstance(k.value,ast.Constant) and k.value.value is False for k in n.keywords):features.add(('frozen parameter construction',))
        self.routing_geometry.sort(key=lambda item:json.dumps(item,sort_keys=True))
        return sorted(features)


def source_key(sources):return json.dumps(sources,sort_keys=True)


@lru_cache(maxsize=1)
def pair_references():
    """Finite semantic reviews are bound to both complete source snapshots."""
    path=Path(__file__).with_name('ontology_addition_pair_references.json')
    rows=json.loads(path.read_text(encoding='utf-8'))['references']
    result={(r['campaign_key'],r['before_sha256'],r['after_sha256']):r['result'] for r in rows}
    from experiments.ontology_addition_b04_references import references as b04_references
    from experiments.ontology_addition_b03_references import references as b03_references
    from experiments.ontology_addition_b03c1_references import references as b03c1_references
    from experiments.ontology_addition_b01c3_references import references as b01c3_references
    from experiments.ontology_addition_b05c3_references import references as b05c3_references
    from experiments.ontology_addition_remaining_references import references as remaining_references
    result.update(b04_references()); result.update(b03_references()); result.update(b03c1_references()); result.update(b01c3_references()); result.update(b05c3_references()); result.update(remaining_references()); return result


def exact_pair_review(before_sources,after_sources,campaign_key):
    key=(campaign_key,hashlib.sha256(source_key(before_sources).encode()).hexdigest(),hashlib.sha256(source_key(after_sources).encode()).hexdigest())
    result=pair_references().get(key)
    if result is None:return None
    result=copy.deepcopy(result)
    result['evidence'].append({'scope':'complete before/after source binding','kind':'SHA-256-bound direct review','before_source_sha256':key[1],'after_source_sha256':key[2]})
    return result


@lru_cache(maxsize=192)
def _profile(encoded):
    try:return SignalReview(json.loads(encoded)).inspect()
    except (SyntaxError,ValueError,TypeError,KeyError,IndexError,AttributeError,RecursionError) as e:
        return {'error':str(e),'error_type':type(e).__name__}


def profile(sources):return _profile(source_key(sources))


def normalization_topology(fingerprint):
    """Return source-traced quotient, partial-coordinate and RMS forms only."""
    forms=[]
    for label in str(fingerprint.get('normalization','')).split(','):
        label=label.strip();lower=label.lower()
        if not (lower.endswith('layernorm') or lower.endswith('rmsnorm')):continue
        if lower.endswith('rmsnorm'):form='rms'
        elif 'quotient' in lower:form='quotient'
        elif 'partial' in lower or 'pruned' in lower:form='partial-coordinate'
        else:continue
        forms.append((label,form))
    return tuple(sorted(set(forms)))


def expression_tags(expr):
    tags=set()
    if isinstance(expr,(tuple,list)):
        if expr and isinstance(expr[0],str):tags.add(expr[0])
        for child in expr:tags.update(expression_tags(child))
    return tags


def last_feature_reductions(expr):
    """Recognize only final-feature mean/sum reductions, not token history."""
    if not isinstance(expr,(tuple,list)) or not expr:return True
    if expr[0] in {'mean','sum'}:
        options=expr[-1] if isinstance(expr[-1],(tuple,list)) else ()
        dimension=next((v for k,v in options if k=='dim'),None) if all(isinstance(v,(tuple,list)) and len(v)==2 for v in options) else None
        if dimension not in {'UnaryOp(op=USub(), operand=Constant(value=1))','Constant(value=-1)'}:return False
    return all(last_feature_reductions(v) for v in expr)


def relative_bucket_layout(sources):
    """Return the directly stored layout of a traced relative-bias table.

    The table's geometry is not determined by its lookup expression alone:
    changing the split between its learned near rows and repeated far rows
    changes which distances share a score coordinate.  Keep the source AST
    spelling of each width so configuration-dependent layouts are never
    guessed to be equal.
    """
    widths = {}
    for source in sources.values():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {dotted(target) for target in targets}
            kind = 'near' if 'self.rel_bias' in names else 'far' if 'self.far_rel_bias' in names else None
            if kind is None:
                continue
            zeros = next((call for call in ast.walk(node.value) if isinstance(call, ast.Call) and dotted(call.func).split('.')[-1] in {'zeros', 'empty', 'ones'} and call.args), None)
            if zeros is None:
                continue
            widths[kind] = ast.unparse(zeros.args[-1])
    if set(widths) == {'near', 'far'}:
        return widths['near'], widths['far']
    return None


def task_fingerprint(sources,p,key):
    """Complete descriptive axes for the fully traced transformer grammar.

    Component descriptions distinguish storage coordinates from the family
    descriptors used for discovery.  No descendant copies a seed fingerprint.
    """
    root=p['root'];roles=p.get('roles',{})
    token=[v for k,v in roles.items() if k.startswith(root+'.') and re.search(r'token_emb|token_embedding|\.embedding$|operand_digit_embedding|generated_digit_embedding',k)]
    position=[v for k,v in roles.items() if k.startswith(root+'.') and re.search(r'pos_emb|position_emb',k)]
    relative=p.get('score_offsets') or [v for k,v in roles.items() if re.search(r'relative_bias|relative_position|rel_bias',k)]
    out=[v for k,v in roles.items() if k.startswith(root+'.') and re.search(r'lm_head|\.output$|output_projection|classifier',k)]
    ttags=set().union(*(expression_tags(v['expression']) for v in token)) if token else set()
    inputs=[v['input_expression'] for k,v in roles.items() if k.startswith(root+'.') and re.search(r'\.block[s]?$|\.attn$|\.attention$',k)]
    if inputs:ttags|=set().union(*(expression_tags(v) for v in inputs))
    signal=set(p['signal_operations']);params=set(p['parameter_operations'])
    active_code='\n'.join(sources.values())
    # The source expression must contain the digit split, min/max canonicalizer
    # and its lookup use. A class name containing "symmetric" is insufficient.
    unordered={'minimum','maximum','FloorDiv','Mod'}<=ttags or ('operand-swap canonical index map',) in p['construction_features']
    if 'lookup' not in ttags:
        if ttags&{'sin','cos','sinc'}:embedding='parametric trigonometric token-feature construction'
        elif ttags&{'Pow','square'} or 'Mult' in ttags and 'stack' in ttags:embedding='parametric polynomial token-feature construction'
        elif 'one_hot' in ttags:embedding='deterministic categorical token-feature construction'
        elif token and ttags&{'scale','offset','cast:float','affine'}:embedding='parametric ordinal-token feature construction'
        else:return {},False,None
    elif unordered:embedding='unordered operand-pair quotient lookup'
    elif {'FloorDiv','Mod'}<=ttags:
        embedding='composed ordered digit lookups'
        def digit_compositions(value):
            if not isinstance(value,tuple):return set()
            found=set().union(*(digit_compositions(v) for v in value))
            if value and value[0] in {'Mult','Add','cat'}:
                child_tags=[expression_tags(v) for v in value[1:] if isinstance(v,tuple)]
                left=any({'lookup','FloorDiv'}<=t and 'Mod' not in t for t in child_tags)
                right=any({'lookup','Mod'}<=t and 'FloorDiv' not in t for t in child_tags)
                if left and right:found.add(value[0])
            return found
        compositions=set().union(*(digit_compositions(v['expression']) for v in token),*(digit_compositions(v) for v in inputs))
        if compositions=={'Mult'}:embedding='multiplicative composition of ordered digit lookups'
        elif compositions=={'cat'}:embedding='concatenated ordered digit lookups'
        elif compositions=={'Add'}:embedding='additive composition of ordered digit lookups'
        elif compositions:embedding='composed ordered digit lookups: '+', '.join(sorted(compositions))
    else:embedding='token lookup table'
    token_types={v['type'] for v in token}
    coefficient_scopes={scope for v in token for scope in v.get('coefficient_scopes',[])}
    coefficient_nonlinear_codes={code for v in token for code in v.get('coefficient_nonlinear_codes',[])}
    table_nonlinear=[v for v in p.get('parameter_nonlinear_sites',[]) if (v['scope'].split('.')[0] in token_types or v['scope'] in coefficient_scopes or v['code'] in coefficient_nonlinear_codes) and v['operator'] in ACT|{'sin','cos','exp','softplus','pow','square'}]
    if table_nonlinear:embedding='nonlinear decoded learned token table'
    factors=any(('Sequential(' in entry['constructor'] and 'Linear(' in entry['constructor']) for entry in token)
    factors=factors or any(v['scope'].split('.')[0] in token_types and v['operator']=='MatMult' for v in p.get('parameter_product_sites',[]))
    embed_detail=('factorized '+embedding) if factors else embedding
    content='softmax' in signal or 'log_softmax' in signal
    position_only=('softmax' in params or 'log_softmax' in params) and not content
    if position_only and p.get('geometry_forms'):relative=relative or p['geometry_forms']
    def contains_equation(value,equation):
        return isinstance(value,tuple) and (value==('literal',equation) or any(contains_equation(v,equation) for v in value))
    direct_attention=contains_equation(p['signal_expression'],'hij,bjk,hok->bio')
    attention=('MatMult' in signal or 'matmul' in signal or direct_attention) and (content or position_only)
    if not attention:return {},False,None
    # The complete template is for tokenwise transformer computation. New
    # explicit integer carry logic or feature-history reductions need their
    # own component review, even when a transition itself can be matched.
    for role,value in roles.items():
        if role.startswith(root+'.') and not re.search(r'token_emb|token_embedding|pos_emb|position_emb',role):
            if expression_tags(value['expression'])&{'FloorDiv','Mod','cumsum','cumprod'}:return {},False,None
            if not last_feature_reductions(value['expression']):return {},False,None
    tags=expression_tags(p['signal_expression'])
    if p.get('effects'):return {},False,None
    enc=p['encodings'].get('encode_inputs')
    encoder=p.get('encoding_descriptors',{}).get('encode_inputs')
    target_codec=p.get('encoding_descriptors',{}).get('encode_targets')
    decoder=p.get('encoding_descriptors',{}).get('decode_targets')
    if 'encode_targets' in p['encodings'] and not target_codec and p['encodings']['encode_targets']!='supplied decimal digit targets; optional constant end token':return {},False,None
    if 'decode_targets' in p['encodings'] and not decoder:return {},False,None
    known_encoding=key=='openevolve_v21' or encoder or enc=='ordered base-10 operand-pair token sequence; unchanged incoming digit order'
    if not known_encoding:return {},False,None
    additive=bool(position or p.get('position_forms'))
    bias_tags=expression_tags(tuple(p.get('score_offset_geometry',[])) + (tuple(p.get('geometry_forms',[])) if position_only else ()))
    bucket_layout=relative_bucket_layout(sources)
    score_bias=('Gaussian-mixture relative-distance score function' if {'logsumexp','square','Sub','position axis'}<=bias_tags and any(v['operator']=='exp' for v in p.get('parameter_nonlinear_sites',[])) else 'quadratic positional pointer with learned centers/sharpness' if {'square','Sub','position axis'}<=bias_tags and any(v['operator'] in {'softplus','exp'} for v in p.get('parameter_nonlinear_sites',[])) else 'relative-distance score table' if 'positional table lookup' in bias_tags else 'input-independent attention-score coefficients' if relative else 'none')
    if bucket_layout:
        score_bias += '; stored near/far row widths '+bucket_layout[0]+' / '+bucket_layout[1]
    position_type=('additive position features' if additive else 'no separate additive position path')+(' + input-independent attention-score bias path' if relative else '')
    if score_bias!='none':position_type+='; '+score_bias
    if p.get('position_forms') and any('one_hot' in x for x in p['position_forms']):position_type+='; discrete one-hot phase coordinates'
    routing='content-dependent softmax QK scores' if content else 'position-only learned softmax scores'
    # A causal mask is explicit in the active attention computation. Its exact
    # AST remains in the signal signature; this label does not erase geometry.
    mask='masked routing' if 'masked_fill' in signal or 'masked_fill' in params else 'unmasked routing'
    geom_ops=set().union(*(set(g['operators']) for g in p.get('routing_geometry',[]))) if p.get('routing_geometry') else set()
    if 'tril' in geom_ops and 'triu' not in geom_ops:mask='lower-triangular causal routing'
    elif 'triu' in geom_ops and 'tril' not in geom_ops:mask='upper-triangular routing'
    mlp=[v for k,v in roles.items() if re.search(r'\.mlp$|\.feedforward$',k)]
    mtags=set().union(*(expression_tags(v['expression']) for v in mlp)) if mlp else set()
    def has_gate(k):
        if not isinstance(k,tuple):return False
        if k and k[0]=='gated-glu':return True
        if len(k)==3 and k[0]=='Mult' and k[1]!=k[2]:
            if any(isinstance(v,tuple) and v and v[0] in {'activation:sigmoid','activation:silu','activation:gelu'} for v in k[1:]):return True
        return any(has_gate(v) for v in k)
    gated=any(has_gate(v['expression']) for v in mlp)
    feedforward='multiplicatively gated pointwise MLP' if gated else 'trigonometric feature basis with affine readout' if mtags&{'sin','cos','sinc'} else 'polynomial feature basis with affine readout' if mtags&{'Pow','square'} and not any(t.startswith('activation:') for t in mtags) else 'ungated pointwise MLP' if mlp else 'no separate pointwise feedforward module'
    acts=sorted({x.split(':',1)[1] for x in expression_tags(p['signal_expression']) if x.startswith('activation:')})
    acts+=sorted(mtags&{'sin','cos','sinc','square','Pow','gated-glu'})
    norms=sorted((signal&NORM)|({x for x in signal if x.lower().endswith('layernorm') or x.lower().endswith('rmsnorm')}))
    norm_modules=tuple(sorted(set(signal)&{'LayerNorm','RMSNorm'}))
    aliases=[]
    for filename,text in sources.items():
        tree=ast.parse(text)
        for n in ast.walk(tree):
            if isinstance(n,ast.Assign) and isinstance(n.value,ast.Attribute) and dotted(n.value).startswith('self.'):
                for target in n.targets:
                    if isinstance(target,ast.Attribute) and target.attr=='weight' and n.value.attr=='weight':aliases.append(ast.unparse(n))
            if isinstance(n,ast.Call) and dotted(n.func)=='object.__setattr__' and len(n.args)==3 and isinstance(n.args[1],ast.Constant) and n.args[1].value in {'embedding','token_embedding','token_emb'}:aliases.append('shared embedding module reference')
    aliases=sorted(set(aliases+p.get('initialization_aliases',[])+[e['kind'] for e in p['evidence'] if e.get('kind','').startswith('shared affine projection repeated over explicit attention heads')]))
    constructions=[]
    if params&{'cat','pad','stack','masked_scatter','scatter','index_copy','index_put'}:constructions.append('stored coordinates assembled by indexing/concatenation/padding')
    if params&{'mean','sum','MatMult','matmul','dot','outer'}:constructions.append('centering or linear coordinate/basis reconstruction')
    if any(x.startswith('learned-') for x in params):constructions.append('composed learned coefficient factors')
    if p.get('parameter_nonlinear_sites'):constructions.append('nonlinear learned-coefficient operations: '+', '.join(sorted({v['operator'] for v in p['parameter_nonlinear_sites']})))
    if p['construction_features']:constructions.extend(' / '.join(map(str,f)) for f in p['construction_features'])
    constructions.extend(p.get('kernel_mechanisms',[]))
    coefficient='learned affine coefficients'+(' and lookup coordinates' if 'lookup' in ttags else '; token features are generated parametrically')+('; '+ '; '.join(constructions) if constructions else '')
    fingerprint={
        'input_units':'operand-pair and generated decimal-digit tokens',
        'input_transform':'ordered base-10 digit-column packing; token alphabet/delimiters are coordinates',
        'embedding':embed_detail,
        'position':position_type,
        'mixing':'causal transformer attention' if 'causal' in mask or position_only and mask=='masked routing' else 'transformer attention',
        'routing':routing+'; '+mask,
        'state':'feedforward hidden activations; no persistent recurrent state in traced methods',
        'feedforward':feedforward,
        'parameter_construction':coefficient,
        'sharing':'; '.join(aliases) if aliases else 'no explicit input/output weight alias; repeated module calls retain their module parameters',
        'normalization':', '.join(norms) if norms else ('explicit centering/scaling arithmetic' if {'mean','rsqrt'}&signal else 'no normalization in traced signal path'),
        'connectivity':'residual attention and pointwise transformation paths' if 'Add' in signal and mlp else 'fixed composition of traced attention/readout paths',
        'aggregation':'per-position attention-mixed features for token prediction',
        'output':'; '.join(p.get('output_forms',[])) or 'affine vocabulary readout'+(' with factorized coordinates' if any('Sequential(' in v['constructor'] for v in out) else ''),
        'symmetry':'operand-swap invariant unordered pair representation' if unordered else 'ordered operand representation; no explicit min/max quotient in the embedding path',
        'conditional_compute':'fixed forward computation; no persistent-state writes or untraced dynamic branch',
        'activation':', '.join(acts) if acts else 'no pointwise activation outside softmax/normalization',
        'stochasticity':'dropout modules (rate is a setting)' if 'Dropout' in signal or 'dropout' in signal else 'no stochastic operator in the traced forward path',
        'bottleneck':'factorized embedding/readout coordinates' if factors or any('Sequential(' in v['constructor'] for v in out) else 'no separate sequential factor bottleneck',
        'iteration':'fixed forward block stack; autoregressive token iteration is inference',
        'other':'all runtime operations are accounted for by the source-traced signal/coordinate graph',
        'operand_encoding':'ordered base-10 digit-column pair tokens',
        'digit_order':'least-significant column first; unchanged input order',
        'carry_representation':'implicit in attention/hidden features; no explicit persistent carry register',
        'attention_scores':routing+'; '+mask+('; '+score_bias if score_bias!='none' else '')+('; '+ '; '.join(p.get('kernel_mechanisms',[])) if p.get('kernel_mechanisms') else ''),
        'projection_relations':'learned affine value/output maps'+(' and Q/K maps' if content else '; no content Q/K score path')+'; stored-coordinate constraints recorded separately',
        'output_factorization':'autoregressive decimal-digit token logits; optional constant termination token'+('; '+ '; '.join(p.get('output_forms',[])) if p.get('output_forms') else ''),
    }
    if encoder:
        for component in ['input_units','input_transform','operand_encoding','digit_order']:fingerprint[component]=encoder[component]
        fingerprint['carry_representation']=encoder['carry_representation']+'; traced model has implicit hidden-state carry and no persistent carry register'
    if target_codec:
        fingerprint['output_factorization']=target_codec['output_factorization']
        fingerprint['input_units']+='; generated target units: '+target_codec['target_units']
        fingerprint['iteration']+='; '+target_codec['iteration']
    if decoder:fingerprint['output_factorization']+='; '+decoder['output_decoding']
    family={'operand_representation':encoder['family'] if encoder else enc or 'ordered base-10 operand-pair token sequence; unchanged incoming digit order','target_representation':(target_codec['family'] if target_codec else 'decimal-digit token targets',decoder['family'] if decoder else 'direct decimal-digit output'),'embedding':embedding,'position':'additive' if additive else 'none','relative_scores':bool(relative),'attention':routing,'feedforward':feedforward,'normalization_modules':norm_modules,'normalization_topology':normalization_topology(fingerprint),'relative_score_layout':relative_bucket_layout(sources),'parameter_generator':[f for f in p['construction_features'] if f and f[0]=='fixed algebraic generator'],'attention_kernel':p.get('kernel_mechanisms',[]),'score_bias':score_bias}
    assert set(CORE+TASK_KEYS['addition'].split())<=fingerprint.keys()
    return fingerprint,True,family


def review_pair(before_sources, after_sources, campaign_key):
    """Return a supported source adjudication, or None for the residual queue."""
    if campaign_key not in {'openevolve_v21','tiny_adderboard_v21'}:return None
    exact=exact_pair_review(before_sources,after_sources,campaign_key)
    if exact is not None:
        # Finite pair adjudications retain their exact transition evidence, but
        # their historical display signatures can predate a source-derived
        # discriminator.  Rehydrate only normalization topology from this
        # candidate's current traced fingerprint; do not rewrite the bound
        # classification, evidence, or any other reviewed signature field.
        after=profile(after_sources)
        if 'error' not in after:
            _,complete,family=task_fingerprint(after_sources,after,campaign_key)
            if complete and family:
                signature=copy.deepcopy(exact.get('family_signature') or {})
                for field in ('normalization_modules','normalization_topology'):
                    signature[field]=copy.deepcopy(family[field])
                exact['family_signature']=signature
        return exact
    before,after=profile(before_sources),profile(after_sources)
    if 'error' in before or 'error' in after:return None
    fp,complete,family=task_fingerprint(after_sources,after,campaign_key)
    beforefp,_,beforefamily=task_fingerprint(before_sources,before,campaign_key)
    # A nonlinear parameter generator is not erased as affine coordinates.
    algebraic={'sin','cos','exp','sigmoid','tanh','softmax','pow','Pow','fft','rfft','irfft','learned-MatMult','learned-Mult','learned-Mul','learned-Div','learned-Pow','conditional coefficient'}
    nonlinear_equal=before.get('parameter_nonlinear_sites')==after.get('parameter_nonlinear_sites')
    algebra_before=algebraic&set(before['parameter_operations']);algebra_after=algebraic&set(after['parameter_operations'])
    product_changes=[v for p,other in [(before,after),(after,before)] for v in p.get('parameter_product_sites',[]) if v not in other.get('parameter_product_sites',[]) and v['operator']=='Mult']
    score_sources='\n'.join(str(p.get('score_forms',[]))+str(p.get('score_offsets',[])) for p in [before,after])
    position_scales=bool(product_changes) and all(any(re.search(r'pos_emb|position_emb',role) for role in v['roles']) or v.get('expanded_code') and v['expanded_code'] in score_sources for v in product_changes)
    coordinate_factors=bool(family and beforefamily and family==beforefamily and (algebra_before^algebra_after)<=({'learned-MatMult','learned-Mult'} if position_scales else {'learned-MatMult'}) and nonlinear_equal)
    def mechanism_features(p):return [f for f in p['construction_features'] if f[0]!='additive position feature generator']
    before_norm_modules=set(before['signal_operations']) & {'LayerNorm','RMSNorm'}
    after_norm_modules=set(after['signal_operations']) & {'LayerNorm','RMSNorm'}
    before_norm_topology=beforefamily.get('normalization_topology') if beforefamily else ()
    after_norm_topology=family.get('normalization_topology') if family else ()
    same_relative_layout=not family or not beforefamily or family.get('relative_score_layout')==beforefamily.get('relative_score_layout')
    matching=(before['signal_signature']==after['signal_signature'] and mechanism_features(before)==mechanism_features(after) and before.get('routing_geometry')==after.get('routing_geometry') and before.get('score_offset_geometry')==after.get('score_offset_geometry') and before_norm_modules==after_norm_modules and before_norm_topology==after_norm_topology and same_relative_layout and (algebra_before==algebra_after or coordinate_factors) and (nonlinear_equal or family and beforefamily and family.get('attention_kernel') and family['attention_kernel']==beforefamily['attention_kernel']))
    witnesses=[]
    if family and beforefamily:
        if family['operand_representation']!=beforefamily['operand_representation']:witnesses.extend(['input_transform','operand_encoding','digit_order'])
        if family['target_representation']!=beforefamily['target_representation']:witnesses.extend(['input_units','output_factorization','iteration'])
        if family['embedding']!=beforefamily['embedding']:witnesses.append('embedding')
        if family['relative_scores']!=beforefamily['relative_scores']:witnesses.extend(['position','attention_scores'])
        if family.get('relative_score_layout')!=beforefamily.get('relative_score_layout'):witnesses.extend(['position','attention_scores'])
        # A fixed algebraic basis that is active in the traced model is part
        # of the representation, even when both versions have an additive
        # position path.  Do not collapse a learned position lookup and a
        # Fourier/basis synthesis merely because they occupy the same axis.
        if family['parameter_generator']!=beforefamily['parameter_generator']:witnesses.extend(['position','parameter_construction'])
        if family['attention_kernel']!=beforefamily['attention_kernel']:witnesses.extend(['parameter_construction','attention_scores'])
        if family['score_bias']!=beforefamily['score_bias'] and any(any(label in f['score_bias'] for label in ['quadratic positional pointer','Gaussian-mixture']) for f in [family,beforefamily]):witnesses.extend(['position','parameter_construction','attention_scores'])
        if family['attention']!=beforefamily['attention']:witnesses.extend(['routing','attention_scores'])
        # A rooted module-level normalization replacement (for example,
        # LayerNorm -> RMSNorm) is a mechanism change.  Keep this narrower
        # than arbitrary functional normalization expressions: those need
        # their own signal proof and cannot be promoted merely from a label.
        if before_norm_modules!=after_norm_modules or before_norm_topology!=after_norm_topology:witnesses.append('normalization')
        if ('gated' in family['feedforward'] and 'ungated' not in family['feedforward'])!=('gated' in beforefamily['feedforward'] and 'ungated' not in beforefamily['feedforward']):witnesses.append('feedforward')
        if family['feedforward']!=beforefamily['feedforward'] and ('feature basis' in family['feedforward'] or 'feature basis' in beforefamily['feedforward']):witnesses.append('feedforward')
        if fp.get('symmetry')!=beforefp.get('symmetry'):witnesses.append('symmetry')
        if ('lower-triangular' in fp.get('routing','') and 'upper-triangular' in beforefp.get('routing','')) or ('upper-triangular' in fp.get('routing','') and 'lower-triangular' in beforefp.get('routing','')):witnesses.extend(['routing','attention_scores'])
    if not matching and not witnesses:return None
    # A family witness can distinguish two source representations while the
    # compact display label stays the same (for example, both paths are
    # additive position features).  Keep that witnessed ontology component in
    # the record: callers must not treat a complete fingerprint alone as a
    # transition decision.
    changed=sorted(set([k for k,v in fp.items() if beforefp.get(k)!=v] if beforefp else [])|set(witnesses))
    changing=not matching and bool(witnesses)
    notes=('Addition signal audit: source-traced mechanism changes in '+', '.join(sorted(set(witnesses)))+'. '+ '; '.join(k+': '+str(beforefp.get(k))+' -> '+str(fp.get(k)) for k in sorted(set(witnesses)))) if changing else 'Addition signal audit: the same rooted input computation and operand representation survive source interpretation. Affine coordinates, token alphabet/delimiter coordinates and training procedure changes preserve this ontology; descriptive component changes remain in the fingerprint.'
    if changing and 'parameter_construction' in witnesses and family.get('attention_kernel')!=beforefamily.get('attention_kernel'):notes+=' The handoff algebraic-construction category applies: learned fractional phase shifts construct head kernels nonlinearly from a shared base, beyond ordinary equality tying or an invertible free basis change.'
    return {'classification':'changing' if changing else 'preserving','fingerprint':fp,'fingerprint_complete':complete,
            'notes':notes,
            'changed_components':changed, 'evidence':after['evidence'],
            'family_signature':family,'reviewer':VERSION,'semantic_signature':after['signal_signature']}





