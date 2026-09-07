"""Recognize affine readout coordinate rewrites without erasing feature changes."""
import ast
import copy
import hashlib
import json
from dataclasses import dataclass

from experiments.ontology_semantics import program, constant, UNKNOWN, SKIP_METHODS
from experiments.ontology_transition_review import NumericSettings, fixed_schedule
from experiments.review_ontology_sources import dotted


@dataclass(frozen=True)
class Affine:
    atoms: frozenset
    projected: bool=False


def combine(items):
    if any(not isinstance(v,Affine) for v in items):raise ValueError('Not affine')
    return Affine(frozenset().union(*(v.atoms for v in items)),any(v.projected for v in items))


class Readout:
    def __init__(self,env,heads):
        self.env=env;self.heads=heads;self.values={};self.arguments=[]

    def expression(self,n):
        if constant(n,self.env) is not UNKNOWN:return Affine(frozenset())
        if isinstance(n,ast.Name) and n.id in self.values:return self.values[n.id]
        if isinstance(n,ast.Subscript):
            if isinstance(n.value,ast.Name) and n.value.id in self.arguments:
                return Affine(frozenset({'argument'+str(self.arguments.index(n.value.id))+'.selected('+ast.unparse(n.slice)+')'}))
            return self.expression(n.value)
        if isinstance(n,(ast.List,ast.Tuple)):return combine([self.expression(x) for x in n.elts])
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):return self.expression(n.operand)
        if isinstance(n,ast.BinOp):
            a,b=self.expression(n.left),self.expression(n.right)
            if isinstance(n.op,(ast.Add,ast.Sub)):return combine([a,b])
            if isinstance(n.op,ast.Mult) and (not a.atoms or not b.atoms):return combine([a,b])
            if isinstance(n.op,ast.Div):
                if not b.atoms:return a
                if not a.projected and not b.projected:
                    return Affine(frozenset({'normalized('+','.join(sorted(a.atoms))+' / '+','.join(sorted(b.atoms))+')'}))
            raise ValueError('Nonlinear arithmetic in readout')
        if isinstance(n,ast.Call):
            name=dotted(n.func);tail=name.split('.')[-1]
            if name in self.heads:
                value=self.expression(n.args[0]);return Affine(value.atoms,True)
            if name in {'torch.cat','torch.concat','torch.stack'}:return self.expression(n.args[0])
            if isinstance(n.func,ast.Attribute):
                value=self.expression(n.func.value)
                if tail in {'view','reshape','flatten','transpose','permute','contiguous','unsqueeze','squeeze'}:return value
                if tail in {'sum','mean','amax','amin','max','min','clamp_min'}:
                    if value.projected:
                        if tail in {'sum','mean'}:return value
                        raise ValueError('Nonlinear post-projection operation')
                    axes=ast.unparse(n.args[0]) if n.args else ','.join(k.arg+'='+ast.unparse(k.value) for k in n.keywords)
                    return Affine(frozenset({tail+'('+','.join(sorted(value.atoms))+';'+axes+')'}))
            raise ValueError('Unrecognized readout call: '+name)
        raise ValueError('Unrecognized readout expression: '+type(n).__name__)

    def run(self,method):
        args=[a.arg for a in method.args.args if a.arg!='self']
        self.arguments=args
        self.values.update({name:Affine(frozenset({'argument'+str(i)})) for i,name in enumerate(args)})
        for n in method.body:
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant):continue
            if isinstance(n,(ast.Assign,ast.AnnAssign)):
                targets=n.targets if isinstance(n,ast.Assign) else [n.target]
                value=constant(n.value,self.env)
                for t in targets:
                    if isinstance(t,(ast.Tuple,ast.List)) and isinstance(n.value,ast.Name) and n.value.id in args:
                        for i,field in enumerate(t.elts):
                            if not isinstance(field,ast.Name):raise ValueError('Complex state unpacking')
                            self.values[field.id]=Affine(frozenset({'argument'+str(args.index(n.value.id))+'.state'+str(i)}))
                    elif isinstance(t,ast.Name):
                        if value is not UNKNOWN:self.env[t.id]=value
                        self.values[t.id]=self.expression(n.value)
                    else:raise ValueError('Readout has state mutation')
            elif isinstance(n,ast.Return):return sorted(self.expression(n.value).atoms)
            else:raise ValueError('Readout contains control flow or side effects')
        raise ValueError('No readout return')


def affine_readout_signature(sources):
    definitions,files,classes,reached,roots,class_envs,env,decisions,trees=program(sources)
    root=next(iter(roots));node=definitions[root]
    builder=definitions.get('build_model')
    if not builder:return None
    root_calls=[n for tree in trees.values() for n in ast.walk(tree) if isinstance(n,ast.Call) and dotted(n.func)==root]
    if len(root_calls)!=1 or not any(n is root_calls[0] for n in ast.walk(next((n for tree in trees.values() for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_model'),ast.Pass()))):
        return None
    method=next((m for m in node.body if isinstance(m,ast.FunctionDef) and m.name=='classify'),None)
    if method is None:return None
    heads={}
    for m in node.body:
        if isinstance(m,ast.FunctionDef) and m.name=='__init__':
            for n in ast.walk(m):
                if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and dotted(n.value.func) in {'nn.Linear','torch.nn.Linear'}:
                    name=dotted(n.targets[0])
                    if 'classifier' in name or 'readout' in name:heads[name]=True
    if not heads:return None
    for m in node.body:
        if isinstance(m,ast.FunctionDef) and m.name not in {'__init__','classify'}|SKIP_METHODS:
            if any(isinstance(n,ast.Attribute) and dotted(n) in heads for n in ast.walk(m)):return None
    try:atoms=Readout(dict(class_envs[root]),heads).run(method)
    except ValueError:return None
    parts=[]
    for name in sorted(reached):
        current=definitions[name]
        if isinstance(current,ast.ClassDef) and name in classes:
            for m in current.body:
                if not isinstance(m,ast.FunctionDef) or m.name in SKIP_METHODS:continue
                if name==root and m.name=='classify':
                    parts.append((name,'affine class readout',atoms));continue
                transformed=copy.deepcopy(m)
                if name==root and m.name=='__init__':
                    # Ignore only the independent affine head declarations.
                    transformed.body=[n for n in transformed.body if not (isinstance(n,ast.Assign) and any(dotted(t) in heads for t in n.targets))]
                if m.name=='frame_schedule' and fixed_schedule(m,class_envs[name]):
                    parts.append((name,m.name,'fixed sampling'));continue
                parts.append((name,m.name,ast.dump(NumericSettings().visit(transformed),include_attributes=False)))
        else:parts.append((name,ast.dump(NumericSettings().visit(copy.deepcopy(current)),include_attributes=False)))
    for name in ['build_model','encode_inputs','encode_targets','decode_targets']:
        if name in definitions:parts.append((name,ast.dump(definitions[name],include_attributes=False)))
    # Keep all referenced non-numerical configuration; unknown globals cannot vanish.
    used={n.id for name in reached for n in ast.walk(definitions[name]) if isinstance(n,ast.Name)}
    for f,tree in trees.items():
        for n in tree.body:
            if isinstance(n,(ast.Assign,ast.AnnAssign)):
                targets=n.targets if isinstance(n,ast.Assign) else [n.target]
                if any(isinstance(t,ast.Name) and t.id in used for t in targets):
                    parts.append((f,ast.dump(NumericSettings().visit(copy.deepcopy(n)),include_attributes=False)))
    return hashlib.sha256(json.dumps(parts,sort_keys=True).encode()).hexdigest()
