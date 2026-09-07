"""Source-only, evidence-preserving normalization for ontology review.

This is a finite Python AST interpreter, not eval/exec and not a semantic oracle.
Only whitelisted constant operations are evaluated. Unknown expressions retain
their syntax; they never silently become an absent mechanism or a numeric size.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import operator
import re

from experiments.review_ontology_sources import dotted

UNKNOWN = object()
BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
       ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
       ast.Is: operator.is_, ast.IsNot: operator.is_not, ast.In: lambda a,b: a in b,
       ast.NotIn: lambda a,b: a not in b}


def constant(node, env):
    """Resolve literals and configuration without running any candidate code."""
    try:
        if node is None:
            return UNKNOWN
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Name, ast.Attribute)):
            key = dotted(node)
            if key in env:
                return env[key]
            if isinstance(node, ast.Attribute):
                base = constant(node.value, env)
                if isinstance(base, dict):
                    return base.get(node.attr, UNKNOWN)
            return UNKNOWN
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = [constant(n, env) for n in node.elts]
            if any(v is UNKNOWN for v in values):
                return UNKNOWN
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Dict):
            keys = [constant(n, env) for n in node.keys]
            values = [constant(n, env) for n in node.values]
            if any(v is UNKNOWN for v in keys+values):
                return UNKNOWN
            return dict(zip(keys, values))
        if isinstance(node, ast.Subscript):
            base, index = constant(node.value, env), constant(node.slice, env)
            return base[index] if base is not UNKNOWN and index is not UNKNOWN else UNKNOWN
        if isinstance(node, ast.UnaryOp):
            value = constant(node.operand, env)
            if value is UNKNOWN:
                return UNKNOWN
            return -value if isinstance(node.op, ast.USub) else +value if isinstance(node.op, ast.UAdd) else not value if isinstance(node.op, ast.Not) else UNKNOWN
        if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            left, right = constant(node.left, env), constant(node.right, env)
            if left is UNKNOWN or right is UNKNOWN:
                return UNKNOWN
            if isinstance(node.op, ast.Mult) and (isinstance(left, (str,list,tuple)) or isinstance(right, (str,list,tuple))):
                return UNKNOWN
            return BINOPS[type(node.op)](left,right)
        if isinstance(node, ast.BoolOp):
            result = UNKNOWN
            for value in node.values:
                result = constant(value, env)
                if result is UNKNOWN:
                    return UNKNOWN
                if isinstance(node.op, ast.And) and not result or isinstance(node.op, ast.Or) and result:
                    break
            return result
        if isinstance(node, ast.Compare):
            values = [constant(n,env) for n in [node.left]+node.comparators]
            if any(v is UNKNOWN for v in values):
                return UNKNOWN
            return all(CMP[type(op)](a,b) for op,a,b in zip(node.ops,values,values[1:]))
        if isinstance(node, ast.IfExp):
            test = constant(node.test,env)
            return UNKNOWN if test is UNKNOWN else constant(node.body if test else node.orelse,env)
        if isinstance(node, ast.Call):
            funcs = {'int':int,'float':float,'bool':bool,'str':str,'len':len,'min':min,'max':max,'abs':abs,
                     'tuple':tuple,'list':list,'dict':dict,'math.sqrt':math.sqrt}
            fn = funcs.get(dotted(node.func))
            args = [constant(n,env) for n in node.args]
            kwargs = {k.arg:constant(k.value,env) for k in node.keywords}
            if fn and all(v is not UNKNOWN for v in args+list(kwargs.values())) and None not in kwargs:
                return fn(*args,**kwargs)
    except (TypeError,ValueError,KeyError,IndexError,ZeroDivisionError,OverflowError):
        pass
    return UNKNOWN


def assign_constants(statements, env):
    for node in statements:
        if isinstance(node, ast.If):
            value = constant(node.test,env)
            if value is not UNKNOWN:
                assign_constants(node.body if value else node.orelse,env)
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            value = constant(node.value,env)
            for target in node.targets if isinstance(node,ast.Assign) else [node.target]:
                key = dotted(target)
                if key and value is not UNKNOWN:
                    env[key] = value


class Specialize(ast.NodeTransformer):
    def __init__(self,env):
        self.env = env
        self.decisions = []

    def visit_If(self,node):
        test = constant(node.test,self.env)
        if test is UNKNOWN:
            return self.generic_visit(node)
        self.decisions.append({'line':node.lineno,'condition':ast.unparse(node.test),'value':bool(test)})
        result = []
        for item in node.body if test else node.orelse:
            value = self.visit(item)
            result.extend(value if isinstance(value,list) else [value] if value else [])
        return result

    def visit_IfExp(self,node):
        test = constant(node.test,self.env)
        return self.generic_visit(node) if test is UNKNOWN else self.visit(node.body if test else node.orelse)

    def visit_Expr(self,node):
        if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self,node):
        previous=self.env
        local=dict(previous)
        # Constructor bindings are known at the selected factory. Other method
        # arguments and locals must not capture constructor variables/globals.
        if node.name!='__init__':
            for arg in node.args.args+node.args.kwonlyargs:local.pop(arg.arg,None)
            for item in ast.walk(node):
                if isinstance(item,ast.Name) and isinstance(item.ctx,ast.Store):local.pop(item.id,None)
        self.env=local
        result=self.generic_visit(node)
        self.env=previous
        return result


def program(sources):
    trees = {f:ast.parse(text) for f,text in sources.items()}
    definitions,files = {},{}
    for f,tree in trees.items():
        for n in tree.body:
            if isinstance(n,(ast.ClassDef,ast.FunctionDef)):
                if n.name in definitions:
                    raise ValueError('Duplicate source-level definition requires module-qualified resolution: '+n.name)
                definitions[n.name],files[n.name] = n,f
    env = {}
    for _ in range(4):
        for tree in trees.values():
            assign_constants(tree.body,env)
        for name,node in definitions.items():
            if isinstance(node,ast.ClassDef):
                attrs = {}
                assign_constants(node.body,attrs)
                env[name] = attrs
    classes = {name for name,n in definitions.items() if isinstance(n,ast.ClassDef) and any(dotted(b).endswith('.Module') for b in n.bases)}
    for _ in range(len(definitions)):
        extended = classes | {name for name,n in definitions.items() if isinstance(n,ast.ClassDef) and any(dotted(b) in classes for b in n.bases)}
        if extended == classes:
            break
        classes = extended
    builder = definitions.get('build_model')
    if builder:
        local = dict(env)
        assign_constants(builder.body,local)
        builder = Specialize(local).visit(copy.deepcopy(builder))
        roots = {dotted(n.func) for n in ast.walk(builder) if isinstance(n,ast.Call)} & classes
    else:
        roots = {'GPT','TinyDecoderLM'} & classes
    if not roots:
        referenced = {n.id for name in classes for n in ast.walk(definitions[name]) if isinstance(n,ast.Name) and n.id!=name}
        roots = classes-referenced
    if len(roots)!=1:
        raise ValueError('Unresolved model roots: '+', '.join(sorted(roots)))
    # Bind factory arguments before resolving forward branches.
    class_envs = {}
    for name in classes:
        ctor = next((n for n in definitions[name].body if isinstance(n,ast.FunctionDef) and n.name=='__init__'),None)
        local = dict(env)
        if ctor:
            args = ctor.args.args
            for arg,value in zip(args[-len(ctor.args.defaults):],ctor.args.defaults):
                resolved = constant(value,env)
                if resolved is not UNKNOWN:
                    local[arg.arg] = resolved
            scope = [builder] if builder and name in roots else list(definitions.values())
            calls = [n for item in scope for n in ast.walk(item) if isinstance(n,ast.Call) and dotted(n.func)==name]
            if len(calls)!=1:
                # Constructor defaults are not evidence when call sites disagree.
                for arg in args[1:]:local.pop(arg.arg,None)
            if len(calls)==1:
                for arg,value in zip(args[1:],calls[0].args):
                    resolved = constant(value,env)
                    if resolved is not UNKNOWN:local[arg.arg]=resolved
                    else:local.pop(arg.arg,None)
                for kw in calls[0].keywords:
                    resolved=constant(kw.value,env)
                    if kw.arg:
                        if resolved is not UNKNOWN:local[kw.arg]=resolved
                        else:local.pop(kw.arg,None)
            assign_constants(ctor.body,local)
        mutable={dotted(n) for m in definitions[name].body if isinstance(m,ast.FunctionDef) and m.name!='__init__'
                 for n in ast.walk(m) if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store)}
        for key in mutable:local.pop(key,None)
        class_envs[name]=local
    specialized,decisions = {},[]
    for name,node in definitions.items():
        transform=Specialize(class_envs.get(name,env))
        specialized[name]=transform.visit(copy.deepcopy(node))
        decisions.extend(dict(d,file=files[name],scope=name) for d in transform.decisions)
    reached=set(roots)
    while True:
        previous=set(reached)
        for name in previous:
            reached.update(n.id for n in ast.walk(specialized[name]) if isinstance(n,ast.Name) and n.id in specialized)
        if reached==previous:
            break
    return specialized,files,classes,reached,roots,class_envs,env,decisions,trees


SIZE_NAMES = re.compile(r'(?:width|hidden|d_model|d_ff|n_head|head_dim|rank|num_layers|n_layer|channels|filters|embed_dim|embedding_dim|sequence_len|max_seq_len|seqLen|head_width|lstmOutput|feature_dim|bottleneck_dim)',re.I)
PRIMITIVES = {'Linear','Embedding','LayerNorm','RMSNorm','BatchNorm1d','BatchNorm2d','GroupNorm','InstanceNorm2d',
              'Conv1d','Conv2d','Conv3d','GRU','GRUCell','LSTM','LSTMCell','RNN','RNNCell',
              'Dropout','Dropout1d','Dropout2d','ReLU','ReLU6','GELU','SiLU','Tanh','Sigmoid','Identity','Flatten',
              'MaxPool1d','MaxPool2d','AvgPool1d','AvgPool2d','AdaptiveAvgPool1d','AdaptiveAvgPool2d'}
COORDINATE_CALLS = {'cat','stack','pad','zeros','zeros_like','ones','ones_like','empty','full','new_zeros',
                    'view','reshape','expand','repeat','unsqueeze','squeeze','transpose','permute','contiguous',
                    'chunk','split','sum','mean','clone','detach','to','float','type_as','narrow','select','flatten',
                    'tril','triu','eye','arange','tensor','as_tensor'}
SKIP_METHODS = {'setup_optimizer','configure_optimizers','init_weights','_init_weights','_initWeights','_initialize_weights',
                '_set_from_full','estimate_flops','num_scaling_params','generate'}


def fixed_frame_schedule(method,env):
    """Prove this hook uses only frame count and configuration, never signal/state."""
    allowed={'range','list','tuple','min','max','round','int','float','len','sorted','set','enumerate'}
    if any(isinstance(n,ast.Call) and dotted(n.func) not in allowed for n in ast.walk(method)):
        return False
    if any(isinstance(n,ast.Attribute) and constant(n,env) is UNKNOWN for n in ast.walk(method)):
        return False
    if any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.Yield,ast.Await)) for n in ast.walk(method)):
        return False
    arguments={a.arg for a in method.args.args}
    locals_={n.id for n in ast.walk(method) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    return all(n.id in arguments|locals_|allowed or n.id in env for n in ast.walk(method) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load))


def fingerprint_program(sources):
    definitions,files,classes,reached,roots,class_envs,env,decisions,trees=program(sources)
    normalized=[]
    coverage=[]
    coordinate_proofs=[]
    leaf={}
    # Only collapse wrappers whose input reaches a standard primitive unchanged.
    for name in reached&classes:
        node=definitions[name]
        fwd=next((n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='forward'),None)
        returns=[n for n in ast.walk(fwd) if isinstance(n,ast.Return)] if fwd else []
        if len(returns)==1 and isinstance(returns[0].value,ast.Call) and len(fwd.args.args)>1:
            call=returns[0].value
            kind={'linear':'Linear','embedding':'Embedding','layer_norm':'LayerNorm','rms_norm':'RMSNorm'}.get(dotted(call.func).split('.')[-1])
            input_name=fwd.args.args[1].arg
            direct_primitive=dotted(call.func).startswith(('F.','torch.nn.functional.','torch.'))
            uses=sum(isinstance(n,ast.Name) and n.id==input_name for n in ast.walk(fwd))
            if kind and direct_primitive and uses==1 and call.args and isinstance(call.args[0],ast.Name) and call.args[0].id==input_name:
                # Weight generators must remain visible; changing their algebra is a separate boundary.
                nonlinear_product=any(isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Mult,ast.Div))
                                      and any(isinstance(v,ast.Attribute) and dotted(v).startswith('self.') for v in ast.walk(n.left))
                                      and any(isinstance(v,ast.Attribute) and dotted(v).startswith('self.') for v in ast.walk(n.right)) for n in ast.walk(fwd))
                if not nonlinear_product and not any(isinstance(n,ast.MatMult) or isinstance(n,ast.Call) and dotted(n.func).split('.')[-1] in {'sin','cos','einsum','matmul','exp','softmax','sigmoid','tanh'} for n in ast.walk(node)):
                    leaf[name]=kind
    for name in sorted(reached):
        node=definitions[name]
        if name in leaf:
            coordinate_proofs.append({'file':files[name],'line':node.lineno,'kind':leaf[name],'scope':name})
            continue
        if isinstance(node,ast.ClassDef) and name in classes:
            params=set()
            modules={}
            buffers=set()
            for n in ast.walk(node):
                if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call):
                    target=dotted(n.targets[0])
                    called=dotted(n.value.func).split('.')[-1]
                    if called=='Parameter':
                        params.add(target)
                    if called in PRIMITIVES or called in classes or called in {'Sequential','ModuleList','ModuleDict'}:
                        modules[target]=leaf.get(called,called)
                if isinstance(n,ast.Call) and dotted(n.func)=='self.register_buffer' and n.args and isinstance(n.args[0],ast.Constant):
                    buffers.add('self.'+str(n.args[0].value))
            for method in node.body:
                if not isinstance(method,ast.FunctionDef) or method.name in SKIP_METHODS:
                    continue
                norm=Normalize(class_envs[name],params,buffers,modules,leaf,classes)
                body=norm.visit(copy.deepcopy(method))
                if method.name=='frame_schedule' and fixed_frame_schedule(method,class_envs[name]):
                    normalized.append((name,method.name,'fixed frame-count-only sampling; indices and count are settings'))
                    coverage.append({'file':files[name],'scope':name+'.'+method.name,'line':method.lineno,
                                     'end_line':method.end_lineno,'proof':'All data dependencies are frame count, literals or resolved configuration.'})
                    continue
                if method.name=='__init__':
                    # Module declarations include hierarchy and categorical switches; raw size and bias settings do not.
                    entries=[]
                    for n in ast.walk(body):
                        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call):
                            if any(isinstance(c,ast.Call) and dotted(c.func).split('.')[-1] in PRIMITIVES|classes|{'Sequential','ModuleList','ModuleDict'} for c in ast.walk(n.value)):
                                entries.append(ast.dump(n,include_attributes=False))
                        if isinstance(n,ast.Assign) and any(dotted(t).startswith('self.') for t in n.targets):
                            key=dotted(n.targets[0])
                            if key in modules or key in params or key in buffers or key.endswith(('.weight','.bias')):
                                continue
                            if SIZE_NAMES.search(key) or re.search(r'(?:dropout|eps|bias)$',key,re.I):
                                continue
                            # Runtime scalars, non-size switches and other state must not disappear.
                            entries.append(ast.dump(n,include_attributes=False))
                        if isinstance(n,ast.Call) and dotted(n.func)=='self.register_buffer':
                            entries.append(ast.dump(n,include_attributes=False))
                    normalized.append((name,'modules',entries))
                else:
                    normalized.append((name,method.name,ast.dump(body,include_attributes=False)))
                coverage.append({'file':files[name],'scope':name+'.'+method.name,'line':method.lineno,
                                 'end_line':method.end_lineno,'operations':sorted({dotted(n.func) for n in ast.walk(method) if isinstance(n,ast.Call)})})
        else:
            # Encoders, custom helper functions and configuration code are retained, not discarded.
            normalized.append((name,ast.dump(Normalize(env,set(),set(),{},leaf,classes).visit(copy.deepcopy(node)),include_attributes=False)))
    for name in ('encode_inputs','encode_targets','decode_targets'):
        if name in definitions and name not in reached:
            normalized.append((name,ast.dump(definitions[name],include_attributes=False)))
    # Model/configuration construction may live in main(), outside build_model.
    # Keep those call sites and assignments feeding them; otherwise a training
    # script can change a categorical model switch without changing the class.
    config_classes={name for name,node in definitions.items() if isinstance(node,ast.ClassDef) and name not in classes and ('config' in name.lower())}
    for filename,tree in trees.items():
        for call in ast.walk(tree):
            if isinstance(call,ast.Call) and dotted(call.func) in roots|config_classes:
                normalized.append(('factory call',filename,ast.dump(Normalize(env,set(),set(),{},leaf,classes|config_classes).visit(copy.deepcopy(call)),include_attributes=False)))
        global_names={n.id for name in reached for n in ast.walk(definitions[name]) if isinstance(n,ast.Name)}
        for statement in tree.body:
            if isinstance(statement,(ast.Assign,ast.AnnAssign)):
                targets=statement.targets if isinstance(statement,ast.Assign) else [statement.target]
                if any(isinstance(t,ast.Name) and t.id in global_names for t in targets):
                    # Preserve referenced globals, including dictionaries and bool/string switches.
                    value=statement.value
                    if not all(isinstance(t,ast.Name) and SIZE_NAMES.search(t.id) for t in targets):
                        normalized.append(('global',filename,ast.dump(statement,include_attributes=False)))
    data=json.dumps(normalized,sort_keys=True)
    return {'shape':hashlib.sha256(data.encode()).hexdigest(),'normalized':normalized,'coverage':coverage,
            'config_decisions':decisions,'coordinate_wrappers':coordinate_proofs,'roots':sorted(roots),
            'resolved_config':{name:{k:v for k,v in local.items() if k.startswith('self.')} for name,local in class_envs.items() if name in reached}}


class Normalize(ast.NodeTransformer):
    def __init__(self,env,params,buffers,modules,leaf,classes):
        self.env=env;self.params=params;self.buffers=buffers;self.modules=modules;self.leaf=leaf;self.classes=classes
        self.parameter_locals=set();self.learned_locals=set();self.size_locals=set();self.symbolic={}

    def learned(self,node):
        return any(isinstance(n,ast.Name) and n.id in self.learned_locals or
                   isinstance(n,ast.Attribute) and (dotted(n) in self.params or dotted(n).startswith('self.') and dotted(n).endswith(('.weight','.bias')))
                   for n in ast.walk(node))

    def parameter_only(self,node):
        if node is None:return True
        if isinstance(node,ast.Constant):return True
        if isinstance(node,ast.Name):return node.id in self.parameter_locals or node.id in self.size_locals or constant(node,self.env) is not UNKNOWN
        if isinstance(node,ast.Attribute):
            key=dotted(node)
            is_weight=key.startswith('self.') and key.endswith(('.weight','.bias'))
            return key in self.params|self.buffers or is_weight or key.endswith(('.shape','.device','.dtype')) or constant(node,self.env) is not UNKNOWN
        if isinstance(node,(ast.Tuple,ast.List)):
            return all(self.parameter_only(v) for v in node.elts)
        if isinstance(node,ast.Subscript):
            return self.parameter_only(node.value) and self.parameter_only(node.slice)
        if isinstance(node,ast.Slice):
            return all(self.parameter_only(v) for v in [node.lower,node.upper,node.step])
        if isinstance(node,ast.UnaryOp):return self.parameter_only(node.operand)
        if isinstance(node,ast.BinOp):
            nonlinear=isinstance(node.op,ast.Mult) and self.learned(node.left) and self.learned(node.right) or isinstance(node.op,ast.Div) and self.learned(node.right)
            return not nonlinear and not isinstance(node.op,(ast.MatMult,ast.Pow)) and self.parameter_only(node.left) and self.parameter_only(node.right)
        if isinstance(node,ast.Call):
            method=dotted(node.func).split('.')[-1]
            receiver=node.func.value if isinstance(node.func,ast.Attribute) and not dotted(node.func).startswith(('torch.','F.')) else None
            return method in COORDINATE_CALLS and (receiver is None or self.parameter_only(receiver)) and all(self.parameter_only(a) for a in node.args+[k.value for k in node.keywords])
        return False

    def visit_FunctionDef(self,node):
        node.returns=None;node.type_comment=None
        for arg in node.args.args+node.args.kwonlyargs:arg.annotation=None
        return self.generic_visit(node)

    def visit_Expr(self,node):
        if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return None
        return self.generic_visit(node)

    def visit_AnnAssign(self,node):
        if node.value is None:return None
        return self.visit(ast.copy_location(ast.Assign(targets=[node.target],value=node.value),node))

    def visit_Assign(self,node):
        targets=[t.id for target in node.targets for t in ast.walk(target) if isinstance(t,ast.Name)]
        if self.parameter_only(node.value):
            if self.learned(node.value):self.learned_locals.update(targets)
            self.parameter_locals.update(targets)
            for t in node.targets:
                if isinstance(t,ast.Name):self.symbolic[t.id]=ast.Constant(value='parameter coordinates')
            # These bindings are substituted wherever consumed. Keep self writes,
            # which can alter runtime state, and tuple unpacking, which carries arity.
            if all(isinstance(t,ast.Name) for t in node.targets):
                return None
        if isinstance(node.value,ast.Attribute) and node.value.attr=='shape':self.size_locals.update(targets)
        return self.generic_visit(node)

    def visit_Call(self,node):
        name=dotted(node.func);tail=name.split('.')[-1]
        if tail in self.leaf:node.func=ast.Name(id=self.leaf[tail],ctx=ast.Load());tail=self.leaf[tail]
        if tail in PRIMITIVES:
            # Operational distinctions retained. Bias, affine scale counts and dimensions are settings.
            keys={'groups','bidirectional','nonlinearity','padding_mode','ceil_mode','count_include_pad','norm_type'}
            kwargs=[]
            for kw in node.keywords:
                if kw.arg in keys:
                    value=constant(kw.value,self.env)
                    if kw.arg=='groups' and isinstance(value,int):value='dense' if value==1 else 'grouped'
                    kwargs.append(ast.keyword(arg=kw.arg,value=ast.Constant(value=value) if isinstance(value,(str,bool,int,float)) else self.visit(kw.value)))
            return ast.Call(func=ast.Name(id=tail,ctx=ast.Load()),args=[],keywords=kwargs)
        if tail in self.classes:
            args=[self.visit(a) for a in node.args if not (type(constant(a,self.env)) in (int,float) or SIZE_NAMES.search(ast.unparse(a)))]
            keywords=[ast.keyword(arg=k.arg,value=self.visit(k.value)) for k in node.keywords if not (isinstance(constant(k.value,self.env),(int,float)) and not isinstance(constant(k.value,self.env),bool) or SIZE_NAMES.search(k.arg or ''))]
            return ast.Call(func=node.func,args=args,keywords=keywords)
        if name in self.modules and self.modules[name] in {'Linear','Embedding','LayerNorm','RMSNorm'}:
            return ast.Call(func=ast.Name(id=self.modules[name],ctx=ast.Load()),args=[self.visit(a) for a in node.args],keywords=[])
        if tail in {'linear','embedding','layer_norm','rms_norm'} and node.args:
            # Discard only proven parameter-only operands of standard primitives.
            rest=node.args[1:]+[k.value for k in node.keywords if k.arg not in {'eps','normalized_shape'}]
            if all(self.parameter_only(a) for a in rest):
                label={'linear':'Linear','embedding':'Embedding','layer_norm':'LayerNorm','rms_norm':'RMSNorm'}[tail]
                return ast.Call(func=ast.Name(id=label,ctx=ast.Load()),args=[self.visit(node.args[0])],keywords=[])
        if tail in {'zeros','ones','empty','new_zeros','new_empty','view','reshape','expand'}:
            args=[]
            for a in node.args:
                value=constant(a,self.env)
                args.append(ast.Constant(value='dimension') if isinstance(value,(int,float)) and abs(value)>3 else self.visit(a))
            return ast.Call(func=self.visit(node.func),args=args,keywords=[self.visit(k) for k in node.keywords])
        return self.generic_visit(node)

    def visit_Name(self,node):
        if isinstance(node.ctx,ast.Load) and node.id in self.symbolic:return self.symbolic[node.id]
        return node


def compare_programs(before,after):
    if before.get('shape') and before['shape']==after.get('shape'):
        return 'preserving','Same computation after configuration resolution and documented parameter/size normalization.'
    return 'uncertain','A residual computation difference requires semantic adjudication.'
