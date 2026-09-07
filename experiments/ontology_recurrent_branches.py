"""Branch-aware extension of the bounded recurrent dataflow reviewer.

Both sides of a runtime branch are retained. This module adds positive semantic
witnesses but never assumes an unmatched branch rewrite is ontology-changing.
"""
import ast
import copy

from experiments.ontology_review_recurrent import Equation, Unsupported, profile, _positive_changes, _readout_features, _recurrent_inputs, _fingerprint, source_sha, _walk, _json, _semantic_graph, CAMPAIGNS
from experiments.ontology_semantics import program
from experiments.review_ontology_sources import dotted


class BranchEquation(Equation):
    def _is_transient_cache_target(self, target):
        """Recognize a deliberately narrow training-only instance cache.

        Some captured KWS candidates save an intermediate readout so the
        module-level training loss can supervise the penultimate frame.  The
        cache is initialized/reset by the model, has a private name, and is
        not fed back into recurrent inference.  It is an observable training
        side effect, so we still trace the right-hand side; accepting it as a
        tensor-state slot would falsely make it persistent recurrent state.
        """
        return (isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == 'self'
                and target.attr.startswith('_'))

    def _record_transient_cache_write(self, target, value):
        # `expr` is intentionally evaluated before accepting the write.  A
        # novel or unaccounted computation cannot be hidden behind a cache.
        traced = self.expr(value)
        if traced not in {('literal', None), ('none',)}:
            self.evidence.append(('transient-training-cache', target.attr, traced))

    def expr(self,n):
        if isinstance(n,ast.Call):
            name=dotted(n.func);tail=name.split('.')[-1]
            if name.startswith('self.') and tail in self.methods:
                if tail in self.stack:raise Unsupported('recursive helper '+tail)
                return BranchEquation(self.modules,self.methods,self.config,self.stack+(tail,),self.attributes).run(self.methods[tail],[self.expr(x) for x in n.args])
        return super().expr(n)

    def run(self,method,args=None):
        self.prepare_local_lists(method)
        self.prepare_local_tensor_writes(method)
        arguments=[a.arg for a in method.args.args if a.arg!='self']
        for i,name in enumerate(arguments):self.values[name]=args[i] if args and i<len(args) else ('state',) if name=='state' else ('signal',i)
        return self.block(method.body)

    def block(self,body):
        for index,n in enumerate(body):
            if isinstance(n,ast.If):
                test=self.expr(n.test);branches=[]
                for statements in [n.body,n.orelse]:
                    other=BranchEquation(dict(self.modules),self.methods,self.config,self.stack,self.attributes)
                    other.values=dict(self.values)
                    other.local_mutables=set(self.local_mutables)
                    other.local_tensor_writes=set(self.local_tensor_writes)
                    branches.append(other.block(statements+body[index+1:]))
                return branches[0] if branches[0]==branches[1] else ('conditional',test,*branches)
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant):continue
            if self.local_list_statement(n):continue
            if isinstance(n,(ast.Assign,ast.AnnAssign)):
                value=self.expr(n.value)
                for t in n.targets if isinstance(n,ast.Assign) else [n.target]:
                    if self._is_transient_cache_target(t):
                        # Reuse the already traced value; do not install it in
                        # `values`, because it is not a local dataflow name.
                        if value not in {('literal', None), ('none',)}:
                            self.evidence.append(('transient-training-cache', t.attr, value))
                    else:self.bind(t,value)
            elif isinstance(n,ast.AugAssign):self.bind(n.target,(type(n.op).__name__,self.expr(n.target),self.expr(n.value)))
            elif isinstance(n,ast.Return):return self.expr(n.value)
            elif isinstance(n,ast.Delete) and all(isinstance(x,ast.Name) for x in n.targets):
                for x in n.targets:self.values.pop(x.id,None)
            elif isinstance(n,ast.Pass):continue
            else:raise Unsupported('statement '+type(n).__name__+' at '+str(n.lineno))
        raise Unsupported('no returned tensor')


def count_schedule_graph(eq):
    """Erase threshold geometry only when the guard uses an incrementing clock."""
    counters=set()
    def returned_components(n):
        if not isinstance(n,tuple):return
        if n[0]=='conditional':
            yield from returned_components(n[2]);yield from returned_components(n[3])
        elif n[0]=='tuple':yield from n[1:]
    for n in returned_components(eq):
        if not isinstance(n,tuple) or len(n)!=3 or n[0]!='Add':continue
        for a,b in [(n[1],n[2]),(n[2],n[1])]:
            if isinstance(a,tuple) and a[0]=='component' and b==('number',1):counters.add(a)
    def clock_only(n):
        if not isinstance(n,tuple):return True
        if n[0]=='component':return n in counters
        if n[0] in {'signal','recurrent-state','recurrent-output','cell-state','attribute','learned-parameter'}:return False
        if n[0] in {'sin','cos','exp','tanh','sigmoid','softmax'}:return False
        return all(clock_only(x) for x in n[1:] if isinstance(x,tuple))
    def visit(n):
        if not isinstance(n,tuple):return n
        if n[0]=='conditional' and clock_only(n[1]):return ('fixed-clock-branch',visit(n[2]),visit(n[3]))
        return (n[0],*(visit(x) if isinstance(x,tuple) else x for x in n[1:]))
    return visit(eq)


def normalize_clock_references(equations):
    step=equations.get('recurrent_step')
    if not step:return equations
    def returned(n):
        if not isinstance(n,tuple):return
        if n[0]=='conditional':yield from returned(n[2]);yield from returned(n[3])
        elif n[0]=='fixed-clock-branch':yield from returned(n[1]);yield from returned(n[2])
        elif n[0]=='tuple':yield from n[1:]
    clocks=set()
    for n in returned(step):
        if not isinstance(n,tuple) or len(n)!=3 or n[0]!='Add':continue
        for a,b in [(n[1],n[2]),(n[2],n[1])]:
            if isinstance(a,tuple) and a[0]=='component' and b==('number',1):clocks.add(a)
    indices={str(c[1]) for c in clocks if c[2]==('state',)}
    def visit(n):
        if not isinstance(n,tuple):return n
        if n[0]=='component' and n in clocks:return ('frame-clock',)
        if len(n)==3 and n[0]=='select' and n[1]==('state',) and n[2] in indices:return ('frame-clock',)
        return (n[0],*(visit(x) if isinstance(x,tuple) else x for x in n[1:]))
    return {name:visit(eq) for name,eq in equations.items()}


def is_plain_sequence_fold(method):
    """Recognize only an exact, chronological fold of recurrent_step.

    No frame transformation, skipped/reversed index, side effect, additional
    accumulator or branch may hide behind this vectorized-path proof.
    """
    names=[a.arg for a in method.args.args if a.arg!='self']
    if len(names)!=2:return False
    frames,state=names
    body=[n for n in method.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
    if len(body)!=2 or not isinstance(body[0],ast.For) or not isinstance(body[1],ast.Return):return False
    loop=body[0]
    if loop.orelse or not isinstance(loop.target,ast.Name) or len(loop.body)!=1:return False
    if not isinstance(body[1].value,ast.Name) or body[1].value.id!=state:return False
    if not isinstance(loop.iter,ast.Call) or dotted(loop.iter.func)!='range' or len(loop.iter.args)!=1 or loop.iter.keywords:return False
    bound=ast.dump(loop.iter.args[0],include_attributes=False)
    allowed={ast.dump(ast.parse(frames+'.shape[1]',mode='eval').body,include_attributes=False),
             ast.dump(ast.parse(frames+'.size(1)',mode='eval').body,include_attributes=False)}
    if bound not in allowed:return False
    update=loop.body[0]
    if not isinstance(update,ast.Assign) or len(update.targets)!=1 or not isinstance(update.targets[0],ast.Name) or update.targets[0].id!=state:return False
    call=update.value
    if not isinstance(call,ast.Call) or dotted(call.func)!='self.recurrent_step' or len(call.args)!=2 or call.keywords:return False
    if not isinstance(call.args[1],ast.Name) or call.args[1].id!=state:return False
    frame=call.args[0]
    if not isinstance(frame,ast.Subscript) or not isinstance(frame.value,ast.Name) or frame.value.id!=frames or not isinstance(frame.slice,ast.Tuple):return False
    axes=frame.slice.elts
    full=lambda x:isinstance(x,ast.Slice) and x.lower is None and x.upper is None and x.step is None
    return len(axes) in {2,3} and full(axes[0]) and isinstance(axes[1],ast.Name) and axes[1].id==loop.target.id and (len(axes)==2 or full(axes[2]))


def branch_profile(sources,campaign):
    p=profile(sources,campaign)
    defs,files,classes,reached,roots,envs,_,_,_=program(sources)
    root=next(iter(roots));methods=p['methods'];modules=dict(p['modules'])
    for name in list(p['errors']):
        if name not in methods:continue
        try:
            p['equations'][name]=BranchEquation(modules,methods,envs[root],attributes=p.get('static_attributes')).run(methods[name]);del p['errors'][name]
        except (Unsupported,ValueError,TypeError,IndexError) as error:p['errors'][name]='branch trace: '+str(error)
    if ('recurrent_sequence' in p['errors'] and 'recurrent_step' in p['equations']
        and is_plain_sequence_fold(methods['recurrent_sequence'])):
        p['equations']['recurrent_sequence']=('causal-fold','time-axis-1',('signal',0),('state',),p['equations']['recurrent_step'])
        del p['errors']['recurrent_sequence']
    raw_step=p['equations'].get('recurrent_step')
    if raw_step:p['equations']['recurrent_step']=count_schedule_graph(raw_step)
    p['equations']=normalize_clock_references(p['equations'])
    step=p['equations'].get('recurrent_step');main=p['equations'].get('classify' if p['task']=='kws' else 'forward')
    if step:
        p['mechanism']['state_statistics']=_readout_features(step)
        p['mechanism']['input_representation']=_recurrent_inputs(step)
    if main:p['mechanism']['readout_features']=_readout_features(main)
    from experiments.ontology_review_recurrent import _exit_observables
    for name in ['exit_mask','should_exit','early_exit']:
        if name in p['equations']:p['mechanism']['exit_policy']=('exit-observables',*_exit_observables(p['equations'][name]));break
    return p


def review_pair(before_sources,after_sources,campaign_key):
    if campaign_key not in CAMPAIGNS or not before_sources or not after_sources:return None
    try:
        before=branch_profile(before_sources,campaign_key);after=branch_profile(after_sources,campaign_key)
        changed=_positive_changes(before,after)
    except (ValueError,TypeError,SyntaxError,RecursionError):return None
    fp,residual=_fingerprint(after)
    if not changed and residual:return None
    return {'classification':'changing' if changed else 'uncertain','fingerprint':fp,'fingerprint_complete':not residual,'residual_components':residual,
      'notes':'A source-traced mechanism difference remains after retaining both runtime branches and abstracting count-only update thresholds.' if changed else 'Both branches and any exact chronological recurrent-step fold are accounted for in the candidate architecture; its transition still needs a preserving or changing proof.',
      'changed_components':sorted(set(x['component'] for x in changed)), 'evidence':list(after['evidence'].values()),'transition_evidence':changed,
      'family_signature':_semantic_graph(after['mechanism']) if not residual else None,'before_source_sha256':source_sha(before_sources),
      'source_sha256':source_sha(after_sources),'reviewer':'recurrent-branch-v1'}
