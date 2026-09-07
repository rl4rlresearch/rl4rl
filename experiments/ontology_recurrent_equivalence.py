"""Same fully accounted recurrent computation modulo documented coordinates."""
import ast
import copy
from experiments.ontology_review_recurrent import profile,_fingerprint,source_sha,CAMPAIGNS,_json,_semantic_graph
from experiments.ontology_semantics import program
from experiments.review_ontology_sources import dotted


def _layout_signature(sources):
    defs,_,classes,reached,_,_,_,_,_=program(sources)
    result=[]
    for cls in sorted(reached&classes):
        for m in defs[cls].body:
            if not isinstance(m,ast.FunctionDef):continue
            for n in ast.walk(m):
                if not isinstance(n,ast.Call):continue
                tail=dotted(n.func).split('.')[-1]
                if tail in {'transpose','permute','movedim','moveaxis','flatten','unflatten','squeeze','unsqueeze','narrow','select','index_select','gather','unfold'}:
                    result.append((m.name,tail,tuple(ast.unparse(x) for x in n.args),tuple((k.arg,ast.unparse(k.value)) for k in n.keywords)))
                if tail in {'view','reshape','expand'}:
                    # Coordinate widths may change; rank/order/symbolic choices
                    # remain. A different view can never silently reorder axes.
                    vals=[]
                    for x in n.args:
                        if isinstance(x,ast.Constant) and type(x.value) in (int,float) and x.value>1:vals.append('dimension')
                        else:vals.append(ast.unparse(x))
                    result.append((m.name,tail,tuple(vals)))
                if tail in {'GRU','GRUCell','LSTM','LSTMCell','RNN','RNNCell','Conv1d','Conv2d'}:
                    result.append((m.name,tail,tuple((k.arg,ast.unparse(k.value)) for k in n.keywords if k.arg in {'batch_first','bidirectional','nonlinearity','padding_mode'})))
    return result


def review_pair(before_sources,after_sources,campaign_key):
    if campaign_key not in CAMPAIGNS:return None
    try:
        before=profile(before_sources,campaign_key);after=profile(after_sources,campaign_key)
        _,a=_fingerprint(before);fp,b=_fingerprint(after)
        if a or b or _json(before['equations'])!=_json(after['equations']) or _json(before['mechanism'])!=_json(after['mechanism']):return None
        if _layout_signature(before_sources)!=_layout_signature(after_sources):return None
    except (ValueError,TypeError,SyntaxError,RecursionError):return None
    return {'classification':'preserving','fingerprint':fp,'fingerprint_complete':True,'residual_components':[],
      'notes':'Every runtime method is accounted for and has the same tensor/state equation after ordinary affine/numerical coordinate normalization; a separate layout/axis guard also matches.',
      'changed_components':[],'evidence':list(after['evidence'].values()),'family_signature':_semantic_graph(after['mechanism']),
      'before_source_sha256':source_sha(before_sources),'source_sha256':source_sha(after_sources),'reviewer':'recurrent-equations-v1'}
