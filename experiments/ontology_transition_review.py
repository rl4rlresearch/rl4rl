"""Additional narrowly scoped transition proofs for numeric settings and schedules."""
import ast
import copy
import hashlib
import json

from experiments.ontology_semantics import program, SKIP_METHODS, constant, UNKNOWN
from experiments.review_ontology_sources import dotted


def fixed_schedule(method,env):
    allowed={'range','list','tuple','min','max','round','int','float','len','sorted','set','enumerate','zip'}
    local={n.id for n in ast.walk(method) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    args={a.arg for a in method.args.args}
    for n in ast.walk(method):
        if isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.Yield,ast.Await)):
            return False
        if isinstance(n,ast.Call):
            name=dotted(n.func)
            if name in allowed:continue
            if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id in local and n.func.attr in {'add','append','extend','discard','remove','sort','reverse','update'}:
                continue
            return False
        if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id not in local|args|allowed and n.id not in env:
            return False
        if isinstance(n,ast.Attribute) and not (isinstance(n.value,ast.Name) and n.value.id in local) and constant(n,env) is UNKNOWN:
            return False
    return True


class NumericSettings(ast.NodeTransformer):
    """Same operations and branches, different nonzero numeric constants.

    Preserve boolean switches, zero, signs, indexing/axes, group topology and
    input/target encoding. This is an ontology rubric, not numeric equivalence.
    """
    def visit_Constant(self,node):
        if type(node.value) in (int,float) and node.value!=0:
            return ast.copy_location(ast.Constant(value='positive scalar' if node.value>0 else 'negative scalar'),node)
        return node

    def visit_Subscript(self,node):
        node.value=self.visit(node.value)
        return node # Index/coordinate selection stays exact.

    def visit_BinOp(self,node):
        # x**1 and x**2 are not the same mechanism with a tuned width.
        return node if isinstance(node.op,ast.Pow) else self.generic_visit(node)

    def visit_Call(self,node):
        # These positional arguments can select axes, partitions, or nonlinear
        # powers. Only keyword dimension protection is insufficient.
        if dotted(node.func).split('.')[-1] in {
            'pow','matrix_power','mean','sum','amax','amin','max','min',
            'chunk','split','tensor_split','unbind','transpose','permute',
            'movedim','moveaxis','select','narrow','index_select','gather',
            'scatter','scatter_add','flip','roll','softmax','log_softmax'}:
            return node
        return self.generic_visit(node)

    def visit_keyword(self,node):
        if node.arg in {'groups','dim','dims','axis','axes','bidirectional','batch_first','is_causal'}:
            return node
        return self.generic_visit(node)

    def visit_Dict(self,node):
        node.values=[v if isinstance(k,ast.Constant) and any(s in str(k.value).lower() for s in ['group','axis','dim','causal','bidirectional']) else self.visit(v) for k,v in zip(node.keys,node.values)]
        return node

    def visit_FunctionDef(self,node):
        node.returns=None
        for arg in node.args.args+node.args.kwonlyargs:arg.annotation=None
        return self.generic_visit(node)


def settings_signature(sources):
    definitions,files,classes,reached,roots,class_envs,env,decisions,trees=program(sources)
    parts=[]
    names=set()
    for name in sorted(reached):
        node=definitions[name]
        if isinstance(node,ast.ClassDef) and name in classes:
            for method in node.body:
                if not isinstance(method,ast.FunctionDef) or method.name in SKIP_METHODS:continue
                if method.name=='frame_schedule' and fixed_schedule(method,class_envs[name]):
                    parts.append((name,method.name,'fixed frame-count-only sampling'))
                else:
                    parts.append((name,method.name,ast.dump(NumericSettings().visit(copy.deepcopy(method)),include_attributes=False)))
                names.update(n.id for n in ast.walk(method) if isinstance(n,ast.Name))
        else:
            parts.append((name,ast.dump(NumericSettings().visit(copy.deepcopy(node)),include_attributes=False)))
            names.update(n.id for n in ast.walk(node) if isinstance(n,ast.Name))
    for name in ['encode_inputs','encode_targets','decode_targets','build_model']:
        if name in definitions:
            node=definitions[name]
            # Encodings cannot be erased as generic numeric tuning.
            parts.append((name,ast.dump(node if name!='build_model' else NumericSettings().visit(copy.deepcopy(node)),include_attributes=False)))
            names.update(n.id for n in ast.walk(node) if isinstance(n,ast.Name))
    for filename,tree in trees.items():
        config_classes={name for name,node in definitions.items() if isinstance(node,ast.ClassDef) and name not in classes and 'config' in name.lower()}
        for call in ast.walk(tree):
            if isinstance(call,ast.Call) and dotted(call.func) in roots|config_classes:
                parts.append(('factory call',filename,ast.dump(NumericSettings().visit(copy.deepcopy(call)),include_attributes=False)))
        for n in tree.body:
            if isinstance(n,(ast.Assign,ast.AnnAssign)):
                targets=n.targets if isinstance(n,ast.Assign) else [n.target]
                if any(isinstance(t,ast.Name) and t.id in names for t in targets):
                    parts.append((filename,ast.dump(NumericSettings().visit(copy.deepcopy(n)),include_attributes=False)))
    return hashlib.sha256(json.dumps(parts,sort_keys=True).encode()).hexdigest()
