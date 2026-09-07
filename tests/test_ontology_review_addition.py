"""Counterexamples for source-only addition review; no candidate is executed."""
from pathlib import Path
from experiments.ontology_review_addition import review_pair, profile, relative_bucket_layout
from experiments.review_ontology_sources import CORE, TASK_KEYS


def source(forward='return self.output(x)', extra='', ctor=''):
    return {'model.py': '''import torch
from torch import nn
from torch.nn import functional as F
class TinyDecoderLM(nn.Module):
 def __init__(self):
  super().__init__()
  self.output=nn.Linear(4,4)
'''+('  '+ctor+'\n' if ctor else '')+' def forward(self,x):\n  '+forward.replace('\n','\n  ')+'\n'+extra}


def test_bias_storage_changes_are_preserving():
    b=source()
    a=source('bias=F.pad(self.bias,(0,1))\nreturn F.linear(x,self.output.weight,bias)',ctor='self.bias=nn.Parameter(torch.zeros(3))')
    assert review_pair(b,a,'openevolve_v21')['classification']=='preserving'


def test_unknown_callable_does_not_silently_become_parameter():
    b=source()
    a=source('return F.linear(x,load_new_weight())')
    assert review_pair(b,a,'openevolve_v21') is None


def test_input_conditioned_weights_are_not_affine_coordinates():
    b=source()
    a=source('return F.linear(x,self.output.weight+x.mean())')
    assert review_pair(b,a,'openevolve_v21') is None


def test_input_signal_erasure_guards():
    b=source()
    for body in ['x[:,0]=0\nreturn self.output(x)',
                 'return self.output(torch.tensor(x))',
                 'self.state=x\nreturn self.output(x)',
                 'return self.output(x*x)',
                 'return self.output(x@x)',
                 'return self.output(1/x)',
                 'return self.output(x.int())',
                 'return self.output(torch.sin(x))']:
        assert review_pair(b,source(body),'openevolve_v21') is None


def test_learned_parameter_generator_is_retained():
    b=source()
    a=source('return F.linear(x,self.output.weight@self.other)',ctor='self.other=nn.Parameter(torch.zeros(4,4))')
    assert review_pair(b,a,'openevolve_v21') is None
    a=source('return F.linear(x,torch.sin(self.output.weight))')
    assert review_pair(b,a,'openevolve_v21') is None
    a=source('return F.linear(x,self.fixed)',ctor="self.register_buffer('fixed',torch.sin(torch.arange(16)).reshape(4,4))")
    assert review_pair(b,a,'openevolve_v21') is None


def test_encoding_edits_are_never_absorbed_as_numeric_settings():
    b=source(extra='def encode_inputs(a,b):\n return a*10+b\n')
    a=source(extra='def encode_inputs(a,b):\n return a+b*10\n')
    assert review_pair(b,a,'tiny_adderboard_v21') is None
    a=source(extra='def encode_inputs(a,b):\n pairs=a*10+b\n return torch.zeros_like(pairs)\n')
    assert review_pair(b,a,'tiny_adderboard_v21') is None
    b=source(extra='def encode_targets(digits):\n return digits\n')
    a=source(extra='def encode_targets(digits):\n return digits[:,:2]\n')
    assert review_pair(b,a,'tiny_adderboard_v21') is None


def test_token_alphabet_relabeling_preserves_operand_representation():
    b=source(extra='def encode_inputs(a,b):\n pairs=10+a*10+b\n return torch.cat((torch.full_like(pairs[:,:1],111),pairs),dim=1)\n')
    a=source(extra='def encode_inputs(a,b):\n pairs=11+a*10+b\n return torch.cat((torch.full_like(pairs[:,:1],112),pairs),dim=1)\n')
    assert review_pair(b,a,'tiny_adderboard_v21')['classification']=='preserving'


def test_feature_coordinate_affine_wrapper_preserves_affine_mechanism():
    b=source()
    a=source('return x[..., :1]*0.2+F.linear(x[...,1:],self.output.weight)')
    assert review_pair(b,a,'openevolve_v21')['classification']=='preserving'
    a=source('return x[:,0]*0.2+F.linear(x[:,1:],self.output.weight)')
    assert review_pair(b,a,'openevolve_v21') is None


def test_unresolved_branch_is_not_chosen_by_defaults():
    b=source()
    a=source('if x.mean()>0:\n return self.output(x)\nreturn self.output(x*x)')
    assert review_pair(b,a,'openevolve_v21') is None


def test_source_is_never_executed(tmp_path):
    marker=tmp_path/'executed'
    b=source(extra=f'import pathlib\npathlib.Path({str(marker)!r}).write_text("executed")\n')
    assert profile(b).get('signal_signature')
    assert not marker.exists()


TRANSFORMER='''import torch
from torch import nn
from torch.nn import functional as F
class Attention(nn.Module):
 def __init__(self):
  super().__init__()
  self.qkv=nn.Linear(6,18)
  self.register_buffer('mask',torch.tril(torch.ones(20,20)))
 def forward(self,x):
  q,k,v=self.qkv(x).chunk(3,dim=-1)
  scores=q@k.transpose(-1,-2)
  scores=scores.masked_fill(self.mask==0,float('-inf'))
  return scores.softmax(dim=-1)@v
class TinyDecoderLM(nn.Module):
 def __init__(self):
  super().__init__()
  self.token_embedding=nn.Embedding(114,6)
  self.position_embedding=nn.Embedding(20,6)
  self.attn=Attention()
  self.output=nn.Linear(6,114)
 def forward(self,idx):
  pos=torch.arange(idx.shape[1])
  hidden=self.token_embedding(idx)+self.position_embedding(pos)
  hidden=hidden+self.attn(hidden)
  return self.output(hidden)
'''


def test_traced_transformer_fingerprint_covers_every_addition_axis():
    src={'model.py':TRANSFORMER}
    r=review_pair(src,src,'openevolve_v21')
    assert r['fingerprint_complete']
    assert set(CORE+TASK_KEYS['addition'].split())<=r['fingerprint'].keys()
    assert all(isinstance(v,str) and v for v in r['fingerprint'].values())
    assert r['family_signature']['embedding']=='token lookup table'


def test_embedding_factorization_and_norm_coordinates_preserve_families():
    b={'model.py':TRANSFORMER}
    a={'model.py':TRANSFORMER.replace('self.token_embedding=nn.Embedding(114,6)','self.token_embedding=nn.Sequential(nn.Embedding(114,3),nn.Linear(3,6))')}
    r=review_pair(b,a,'openevolve_v21')
    assert r['classification']=='preserving'
    assert r['fingerprint']['embedding']=='factorized token lookup table'
    b=source('return self.output(F.layer_norm(x,(4,)))')
    a=source('return self.output(F.layer_norm(x,(4,))*self.scale+self.bias)',ctor='self.scale=nn.Parameter(torch.ones(4)); self.bias=nn.Parameter(torch.zeros(4))')
    assert review_pair(b,a,'openevolve_v21')['classification']=='preserving'


def test_rooted_layernorm_to_rmsnorm_is_a_complete_changing_witness():
    before = {
        'model.py': TRANSFORMER.replace(
            'self.attn=Attention()',
            'self.attn=Attention()\n  self.norm=nn.LayerNorm(6)',
        ).replace(
            'hidden=hidden+self.attn(hidden)',
            'hidden=self.norm(hidden+self.attn(hidden))',
        )
    }
    rms = '''class RMSNorm(nn.Module):
 def forward(self,x):
  return x*torch.rsqrt(x.square().mean(dim=-1,keepdim=True)+1e-5)
'''
    after = {
        'model.py': before['model.py'].replace(
            'class Attention', rms+'class Attention',
        ).replace('self.norm=nn.LayerNorm(6)', 'self.norm=RMSNorm()')
    }
    result = review_pair(before, after, 'openevolve_v21')
    assert result['classification'] == 'changing'
    assert result['fingerprint_complete'] is True
    assert 'normalization' in result['changed_components']
    assert result['family_signature']['normalization_modules'] == ('RMSNorm',)


def test_relative_score_bucket_layout_is_source_bound_and_changing():
    with_buckets = TRANSFORMER.replace(
        "self.register_buffer('mask',torch.tril(torch.ones(20,20)))",
        """self.register_buffer('mask',torch.tril(torch.ones(20,20)))
  self.rel_bias=nn.Parameter(torch.zeros(1,8))
  self.far_rel_bias=nn.Parameter(torch.zeros(7))""",
    ).replace(
        'scores=q@k.transpose(-1,-2)',
        '''scores=q@k.transpose(-1,-2)
  positions=torch.arange(x.shape[1])
  lag=(positions[:,None]-positions[None,:]).clamp_min(0)
  table=torch.cat((self.rel_bias,self.far_rel_bias.expand(1,7)),dim=1)
  scores=scores+F.pad(table,(1,0))[:,lag]''',
    )
    before = {'model.py': with_buckets}
    after = {'model.py': with_buckets.replace('torch.zeros(1,8)', 'torch.zeros(1,7)').replace('expand(1,7)', 'expand(1,8)')}
    assert relative_bucket_layout(before) != relative_bucket_layout(after)
    result = review_pair(before, after, 'openevolve_v21')
    assert result['classification'] == 'changing'
    assert result['fingerprint_complete'] is True
    assert {'position', 'attention_scores'} <= set(result['changed_components'])


def test_fixed_fourier_position_generator_is_a_complete_changing_witness():
    before = {'model.py': TRANSFORMER}
    after = {
        'model.py': TRANSFORMER.replace(
            'self.position_embedding=nn.Embedding(20,6)',
            '''self.position_mix=nn.Linear(6,6,bias=False)
  inv_freq=torch.exp(torch.arange(0,6,2)*(-torch.log(torch.tensor(10000.0))/6))
  self.register_buffer('inv_freq',inv_freq)''',
        ).replace(
            'hidden=self.token_embedding(idx)+self.position_embedding(pos)',
            '''angles=pos.unsqueeze(-1)*self.inv_freq
  pos_features=torch.stack((angles.sin(),angles.cos()),dim=-1).flatten(-2)
  hidden=self.token_embedding(idx)+self.position_mix(pos_features)''',
        )
    }
    result = review_pair(before, after, 'openevolve_v21')
    assert result['classification'] == 'changing'
    assert result['fingerprint_complete'] is True
    assert {'position', 'parameter_construction'} <= set(result['changed_components'])
    assert result['family_signature']['parameter_generator'] == [('fixed algebraic generator', ('exp', 'log'))]


def test_b04_c3_p70_fourier_position_transition_is_source_bound():
    from experiments.ontology_review_addition import pair_references
    key = (
        'openevolve_v21',
        'cf29c502fa94b8f43a5e72c44abedeb91f9571b7288a143ca6e0db73298806ed',
        'abfc843e58615d029bbea0cb33c2078995a33d8c0b614a288a13ecd84a07b854',
    )
    result = pair_references()[key]
    assert result['classification'] == 'changing'
    assert result['fingerprint_complete'] is True
    assert {'position', 'parameter_construction'} <= set(result['changed_components'])
    assert any(e['kind'] == 'directly source-adjudicated fixed Fourier position representation' for e in result['evidence'])


def test_b04_c3_shared_kv_compaction_group_is_source_bound_and_preserving():
    from experiments.ontology_review_addition import pair_references
    keys = [
        ('dc3ecd15bc7ab56333437a5cde2aab99bc65e4c37c0f6ce3e03f50e99f16de5b',
         '0b893e177b5918d08ca78f3f538789b020976d6d3c6cc5f9d7481267c9eb348a'),
        ('2d453b148089c19aa62c8e9146a22fea136b70b346caec8b48471a9bd0bdf2e2',
         '5b2595d615786c00e5b740a53e576450ff58661dd3c249a11d9da73ed1eb08c3'),
        ('2d453b148089c19aa62c8e9146a22fea136b70b346caec8b48471a9bd0bdf2e2',
         'c4531ecdd2db19b30c7530b33577ad9d5baa0695699cca4e02bc5e633c8ab850'),
        ('2d453b148089c19aa62c8e9146a22fea136b70b346caec8b48471a9bd0bdf2e2',
         '7ee5d20e1076ccd2c7f11a582cb5d4dde453e51f5381efa4fe3509dd1f82b113'),
    ]
    for before, after in keys:
        result = pair_references()[('openevolve_v21', before, after)]
        assert result['classification'] == 'preserving'
        assert result['fingerprint_complete'] is True
        assert result['family_signature']['attention'] == 'content-dependent softmax QK scores'
        assert any(e['kind'] == 'directly source-adjudicated shared K/V compaction' for e in result['evidence'])


def test_attention_mask_construction_does_not_disappear():
    b={'model.py':TRANSFORMER}
    a={'model.py':TRANSFORMER.replace('torch.tril','torch.triu')}
    r=review_pair(b,a,'openevolve_v21')
    assert r is None or r['classification']=='changing'


def test_composed_digit_lookup_is_a_positive_boundary_witness():
    b={'model.py':TRANSFORMER}
    custom='''class Digits(nn.Module):
 def __init__(self):
  super().__init__()
  self.left=nn.Embedding(10,3)
  self.right=nn.Embedding(10,3)
 def forward(self,idx):
  return torch.cat((self.left(idx//10),self.right(idx%10)),dim=-1)
'''
    a={'model.py':TRANSFORMER.replace('class Attention',custom+'class Attention').replace('self.token_embedding=nn.Embedding(114,6)','self.token_embedding=Digits()')}
    r=review_pair(b,a,'openevolve_v21')
    assert r['classification']=='changing'
    assert 'embedding' in r['changed_components']
    assert r['fingerprint_complete']


def test_custom_module_retains_all_positional_and_keyword_signals():
    helper='''class Wrapper(nn.Module):
 def forward(self,affine_x,normalized_x):
  return F.linear(normalized_x,self.weight)
'''
    b=source('return self.wrapper(x,x)',extra=helper,ctor='self.wrapper=Wrapper()')
    a=source('return self.wrapper(x,x*x)',extra=helper,ctor='self.wrapper=Wrapper()')
    assert 'error' not in profile(b)
    assert review_pair(b,a,'openevolve_v21') is None
    a=source('return self.wrapper(x,normalized_x=x*x)',extra=helper,ctor='self.wrapper=Wrapper()')
    assert review_pair(b,a,'openevolve_v21') is None
    a=source('return F.linear(input=x,weight=x*x)')
    assert review_pair(source(),a,'openevolve_v21') is None


def test_property_module_alias_and_nested_helper_dispatch():
    helper='''class Embed(nn.Embedding):
 def full_weight(self):
  return self.weight
class Head(nn.Module):
 def __init__(self,embedding):
  object.__setattr__(self,'_embedding',embedding)
 @property
 def embedding(self):
  return object.__getattribute__(self,'_embedding')
 def full_weight(self):
  return self.embedding.full_weight()
 def forward(self,x):
  return F.linear(x,self.full_weight())
'''
    b=source()
    a=source('return self.head(x)',extra=helper,ctor='self.embedding=Embed(4,4); self.head=Head(self.embedding)')
    assert review_pair(b,a,'openevolve_v21')['classification']=='preserving'


def test_precomputed_unordered_embedding_map_is_not_an_ordered_table():
    b={'model.py':TRANSFORMER}
    helper='''class Unordered(nn.Module):
 def __init__(self):
  super().__init__()
  self.weight=nn.Parameter(torch.zeros(70,6))
  ids=torch.arange(100)
  left=ids//10
  right=ids%10
  lo=torch.minimum(left,right)
  hi=torch.maximum(left,right)
  token_map=torch.arange(114)
  token_map[10:110]=hi*(hi+1)//2+lo
  self.register_buffer('token_map',token_map)
 def forward(self,idx):
  return F.embedding(self.token_map[idx],self.weight)
'''
    a={'model.py':TRANSFORMER.replace('class Attention',helper+'class Attention').replace('self.token_embedding=nn.Embedding(114,6)','self.token_embedding=Unordered()')}
    r=review_pair(b,a,'openevolve_v21')
    assert r['classification']=='changing'
    assert r['fingerprint']['embedding']=='unordered operand-pair quotient lookup'
    assert 'operand-swap invariant' in r['fingerprint']['symmetry']


def test_direct_encoder_reference_integrates_into_full_fingerprint():
    import ast
    from experiments.ontology_review_addition_encodings import references
    refs=references()
    separate=next(v for v in refs.values() if v['family'].startswith('separate decimal'))
    body=separate['evidence'][0]['source']
    baseline='def encode_inputs(a,b):\n return a*10+b\n'
    b={'model.py':TRANSFORMER+baseline}
    a={'model.py':TRANSFORMER+body+'\n'}
    r=review_pair(b,a,'tiny_adderboard_v21')
    assert r['classification']=='changing'
    assert r['fingerprint_complete']
    assert r['fingerprint']['input_units']==separate['input_units']
    assert 'operand_encoding' in r['changed_components']


def test_output_subset_coordinates_and_prototype_softmax_gauge():
    b={'model.py':TRANSFORMER}
    subset='''class Subset(nn.Module):
 def __init__(self):
  self.weight=nn.Parameter(torch.zeros(11,6))
  self.register_buffer('ids',torch.arange(11))
 def forward(self,x):
  logits=F.linear(x,self.weight)
  full=logits.new_full((1,1,114),float('-inf'))
  return full.index_copy(-1,self.ids,logits)
'''
    a={'model.py':TRANSFORMER.replace('class Attention',subset+'class Attention').replace('self.output=nn.Linear(6,114)','self.output=Subset()')}
    r=review_pair(b,a,'openevolve_v21')
    assert r['classification']=='preserving'
    assert 'vocabulary subset' in r['fingerprint']['output']
    prototype='''class Prototype(nn.Module):
 def __init__(self):
  self.prototypes=nn.Parameter(torch.zeros(114,6))
 def forward(self,x):
  differences=x.unsqueeze(-2)-self.prototypes
  return -differences.square().sum(dim=-1)
'''
    a={'model.py':TRANSFORMER.replace('class Attention',prototype+'class Attention').replace('self.output=nn.Linear(6,114)','self.output=Prototype()')}
    r=review_pair(b,a,'openevolve_v21')
    assert r['classification']=='preserving'
    assert 'prototype' in r['fingerprint']['output']
    wrong={'model.py':a['model.py'].replace('sum(dim=-1)','sum(dim=-2)')}
    assert review_pair(b,wrong,'openevolve_v21') is None


def test_inherited_coefficient_method_and_fixed_index_masks_are_followed():
    helper='''class Base(nn.Module):
 def weight_matrix(self):
  ids=torch.arange(16)
  free=(ids[:,None]!=self.fixed_indices[None,:]).all(dim=1)
  return self.coordinates.new_zeros(16).scatter(0,ids[free],self.coordinates).view(4,4)
class Projection(Base):
 def forward(self,x):
  return F.linear(x,self.weight_matrix())
'''
    a=source('return self.projection(x)',extra=helper,ctor='self.projection=Projection()')
    assert review_pair(source(),a,'openevolve_v21')['classification']=='preserving'


def test_relative_score_table_coordinates_preserve_index_geometry():
    position_only=TRANSFORMER.replace('q,k,v=self.qkv(x).chunk(3,dim=-1)\n  scores=q@k.transpose(-1,-2)', '''q,k,v=self.qkv(x).chunk(3,dim=-1)
  pos=torch.arange(x.shape[1])
  lag=(pos[:,None]-pos[None,:]).clamp_min(0)
  table=self.bias_table
  scores=F.embedding(lag,table).permute(2,0,1)''')
    b={'model.py':position_only}
    a={'model.py':position_only.replace('table=self.bias_table','table=torch.cat((self.coordinates,self.coordinates.new_zeros(1,2)),dim=0)')}
    assert review_pair(b,a,'openevolve_v21')['classification']=='preserving'
    a={'model.py':position_only.replace('pos[:,None]-pos[None,:]','pos[:,None]+pos[None,:]')}
    r=review_pair(b,a,'openevolve_v21')
    assert r is None or r['classification']=='changing'
    a={'model.py':position_only.replace('scores.softmax(dim=-1)','scores.softmax(dim=-2)')}
    assert review_pair(b,a,'openevolve_v21') is None


def test_fixed_runtime_score_masks_are_retained():
    t=TRANSFORMER.replace('scores=q@k.transpose(-1,-2)','scores=self.fixed_scores').replace('scores=scores.masked_fill(self.mask==0', 'scores=scores.masked_fill(torch.tril(torch.ones(20,20))==0')
    b={'model.py':t};a={'model.py':t.replace('torch.tril(torch.ones(20,20))==0','torch.triu(torch.ones(20,20))==0')}
    r=review_pair(b,a,'openevolve_v21')
    assert r is None or r['classification']=='changing'


def test_super_and_indexed_custom_calls_keep_second_input():
    helper='''class Base(nn.Module):
 def forward(self,x,y):
  return F.linear(y,self.weight)
class Wrapper(Base):
 def forward(self,x,y):
  return super().forward(x,y)
'''
    b=source('return self.wrapper(x,x)',extra=helper,ctor='self.wrapper=Wrapper()')
    a=source('return self.wrapper(x,x*x)',extra=helper,ctor='self.wrapper=Wrapper()')
    assert review_pair(b,a,'openevolve_v21') is None
    b=source('return self.layers[0](x,x)',extra=helper,ctor='self.layers=nn.ModuleList([Wrapper()])')
    a=source('return self.layers[0](x,x*x)',extra=helper,ctor='self.layers=nn.ModuleList([Wrapper()])')
    assert review_pair(b,a,'openevolve_v21') is None


def test_nonlinear_spectral_kernel_introduction_has_direct_source_witness():
    before='''class Attention(nn.Module):
 def __init__(self):
  self.value=nn.Linear(6,6)
  self.relative_bias=nn.Parameter(torch.zeros(2,19))
  self.register_buffer('mask',torch.tril(torch.ones(20,20)))
 def forward(self,x):
  pos=torch.arange(x.shape[1])
  lag=(pos[:,None]-pos[None,:]).clamp_min(0)
  lag_bias=torch.cat((self.relative_bias,self.relative_bias.new_zeros(2,1)),dim=-1)
  scores=lag_bias[:,lag]
  scores=scores.masked_fill(self.mask==0,float('-inf'))
  return scores.softmax(dim=-1)@self.value(x)
'''
    after=before.replace('self.relative_bias=nn.Parameter(torch.zeros(2,19))','self.relative_bias=nn.Parameter(torch.zeros(19))\n  self.relative_shift=nn.Parameter(torch.zeros(1))\n  self.relative_log_scale=nn.Parameter(torch.zeros(1))')
    after=after.replace('lag_bias=torch.cat((self.relative_bias,self.relative_bias.new_zeros(2,1)),dim=-1)', '''base=torch.cat((self.relative_bias,self.relative_bias.new_zeros(1)))
  frequency=torch.arange(base.numel()//2+1)
  shift=torch.cat((self.relative_shift.new_zeros(1),self.relative_shift))
  phase=torch.exp(-2j*3.14159*shift[:,None]*frequency[None,:]/base.numel())
  lag_bias=torch.fft.irfft(torch.fft.rfft(base).unsqueeze(0)*phase,n=base.numel(),dim=-1)
  lag_bias=lag_bias*self.relative_log_scale.exp()''')
    tail=TRANSFORMER[TRANSFORMER.index('class TinyDecoderLM'):]
    b={'model.py':'import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n'+before+tail}
    a={'model.py':'import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n'+after+tail}
    r=review_pair(b,a,'openevolve_v21')
    assert r['classification']=='changing' and r['fingerprint_complete']
    assert 'algebraic-construction category' in r['notes']
    assert 'parameter_construction' in r['changed_components']
    coordinate={'model.py':a['model.py'].replace('  lag_bias=lag_bias*self.relative_log_scale.exp()\n','')}
    assert review_pair(a,coordinate,'openevolve_v21')['classification']=='preserving'


def test_functional_embedding_and_feature_norm_have_complete_components():
    t=TRANSFORMER.replace('self.token_embedding=nn.Embedding(114,6)','self.token_weight=nn.Parameter(torch.zeros(114,6))').replace('self.token_embedding(idx)','F.embedding(idx,self.token_weight)')
    norm='''class FeatureNorm(nn.Module):
 def forward(self,x):
  return x*torch.rsqrt(x.square().mean(dim=-1,keepdim=True)+1e-5)
'''
    t=t.replace('class Attention',norm+'class Attention').replace('self.output=nn.Linear(6,114)','self.output=nn.Linear(6,114)\n  self.final_norm=FeatureNorm()').replace('return self.output(hidden)','return self.output(self.final_norm(hidden))')
    src={'model.py':t};r=review_pair(src,src,'openevolve_v21')
    assert r['fingerprint_complete']
    assert r['fingerprint']['embedding']=='token lookup table'
    history={'model.py':t.replace('mean(dim=-1,keepdim=True)','mean(dim=1,keepdim=True)')}
    r=review_pair(history,history,'openevolve_v21')
    assert r is None or not r['fingerprint_complete']


def test_linear_table_factors_preserve_but_nonlinear_decoder_changes():
    table='''class Table(nn.Module):
 def __init__(self):
  self.weight=nn.Parameter(torch.zeros(114,3))
  self.factor=nn.Parameter(torch.zeros(3,6))
 def full_weight(self):
  return self.weight@self.factor
 def forward(self,idx):
  return F.embedding(idx,self.full_weight())
'''
    b={'model.py':TRANSFORMER}
    t=TRANSFORMER.replace('class Attention',table+'class Attention').replace('self.token_embedding=nn.Embedding(114,6)','self.token_embedding=Table()')
    a={'model.py':t}
    r=review_pair(b,a,'openevolve_v21')
    assert r['classification']=='preserving' and r['fingerprint_complete']
    nonlin={'model.py':t.replace('return self.weight@self.factor','return 2*F.gelu(self.weight@self.factor)')}
    r=review_pair(a,nonlin,'openevolve_v21')
    assert r['classification']=='changing' and r['fingerprint_complete']
    assert 'nonlinear decoded' in r['fingerprint']['embedding']


def test_explicit_head_axis_sum_is_distinct_from_history_sum():
    t=TRANSFORMER.replace('self.qkv=nn.Linear(6,18)','self.qkv=nn.Linear(6,18)\n  self.n_head=2').replace('scores=q@k.transpose(-1,-2)', '''q=q.unsqueeze(2).expand(-1,-1,self.n_head,-1).transpose(1,2)
  k=k.unsqueeze(2).expand(-1,-1,self.n_head,-1).transpose(1,2)
  v=v.unsqueeze(2).expand(-1,-1,self.n_head,-1).transpose(1,2)
  scores=q@k.transpose(-1,-2)''').replace('return scores.softmax(dim=-1)@v', '''att=scores.softmax(dim=-1)
  y=att@v
  return y.sum(dim=1)''')
    src={'model.py':t}
    r=review_pair(src,src,'openevolve_v21')
    assert r['fingerprint_complete']
    for wrong in [t.replace('y.sum(dim=1)','y.sum(dim=2)'),t.replace('.transpose(1,2)','.transpose(0,2)')]:
        src={'model.py':wrong};r=review_pair(src,src,'openevolve_v21')
        assert r is None or not r['fingerprint_complete']


def test_quadratic_position_pointer_is_a_new_parameter_construction():
    t=TRANSFORMER.replace('self.qkv=nn.Linear(6,18)','self.qkv=nn.Linear(6,18)\n  self.bias=nn.Parameter(torch.zeros(20))\n  self.center=nn.Parameter(torch.zeros(2))\n  self.sharpness=nn.Parameter(torch.zeros(2))')
    t=t.replace('scores=q@k.transpose(-1,-2)', '''pos=torch.arange(x.shape[1])
  lag=(pos[:,None]-pos[None,:]).clamp_min(0)
  bias=self.bias[lag]
  scores=q@k.transpose(-1,-2)+bias''')
    before={'model.py':t}
    after={'model.py':t.replace('bias=self.bias[lag]','bias=-F.softplus(self.sharpness)*(pos[None,:]-self.center-pos[:,None]).square()')}
    r=review_pair(before,after,'openevolve_v21')
    assert r['classification']=='changing' and r['fingerprint_complete']
    assert 'quadratic positional pointer' in r['fingerprint']['attention_scores']


def test_cached_relative_lags_keep_direction_and_prefix_geometry():
    t=TRANSFORMER.replace('self.qkv=nn.Linear(6,18)',"self.qkv=nn.Linear(6,18)\n  self.relative_bias=nn.Parameter(torch.zeros(20))")
    t=t.replace('scores=q@k.transpose(-1,-2)', '''positions=torch.arange(x.shape[1])
  lag=(positions[:,None]-positions[None,:]).clamp_min(0)
  scores=q@k.transpose(-1,-2)+self.relative_bias[lag]''')
    cached=t.replace('self.relative_bias=nn.Parameter(torch.zeros(20))', '''self.relative_bias=nn.Parameter(torch.zeros(20))
  positions=torch.arange(20)
  lag=(positions[:,None]-positions[None,:]).clamp_min(0)
  self.register_buffer('lag',lag)''').replace('  positions=torch.arange(x.shape[1])\n  lag=(positions[:,None]-positions[None,:]).clamp_min(0)', '  lag=self.lag[:x.shape[1],:x.shape[1]]')
    b={'model.py':t};a={'model.py':cached}
    assert review_pair(b,a,'openevolve_v21')['classification']=='preserving'
    for wrong in [cached.replace('positions[:,None]-positions[None,:]','positions[None,:]-positions[:,None]'),cached.replace('self.lag[:x.shape[1],:x.shape[1]]','self.lag[1:x.shape[1],:x.shape[1]]')]:
        r=review_pair(b,{'model.py':wrong},'openevolve_v21')
        assert r is None or r['classification']=='changing'


def test_direct_einsum_attention_and_explicit_head_output_have_complete_fingerprints():
    before='''class Attention(nn.Module):
 def __init__(self):
  self.bias=nn.Parameter(torch.zeros(20,2))
  self.weight=nn.Parameter(torch.zeros(2,6,6))
 def forward(self,x):
  positions=torch.arange(x.shape[1])
  lag=(positions[:,None]-positions[None,:]).clamp_min(0)
  scores=self.bias[lag].permute(2,0,1)
  probabilities=scores.softmax(dim=-1)
  return torch.einsum('hij,bjk,hok->bio',probabilities,x,self.weight)
'''
    tail=TRANSFORMER[TRANSFORMER.index('class TinyDecoderLM'):]
    t='import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n'+before+tail
    src={'model.py':t};assert review_pair(src,src,'openevolve_v21')['fingerprint_complete']
    t=t.replace("return torch.einsum('hij,bjk,hok->bio',probabilities,x,self.weight)","mixed=probabilities.unsqueeze(0)@x.unsqueeze(1)\n  return torch.einsum('bhli,hoi->bhlo',mixed,self.weight).sum(dim=1)")
    src={'model.py':t};assert review_pair(src,src,'openevolve_v21')['fingerprint_complete']
    wrong={'model.py':t.replace('sum(dim=1)','sum(dim=2)')};r=review_pair(wrong,wrong,'openevolve_v21')
    assert r is None or not r['fingerprint_complete']


def test_gaussian_mixture_replaces_free_lag_table():
    t=TRANSFORMER.replace('self.qkv=nn.Linear(6,18)', '''self.qkv=nn.Linear(6,18)
  self.bias=nn.Parameter(torch.zeros(20))
  self.center=nn.Parameter(torch.zeros(3))
  self.precision=nn.Parameter(torch.zeros(3))
  self.mix=nn.Parameter(torch.zeros(3))''').replace('scores=q@k.transpose(-1,-2)', '''pos=torch.arange(x.shape[1])
  lag=(pos[:,None]-pos[None,:]).clamp_min(0)
  scores=self.bias[lag]''')
    b={'model.py':t};a={'model.py':t.replace('scores=self.bias[lag]', 'components=self.mix-self.precision.exp()*(lag[:,:,None]-self.center).square()\n  scores=torch.logsumexp(components,dim=-1)')}
    r=review_pair(b,a,'openevolve_v21')
    assert r['classification']=='changing' and r['fingerprint_complete']
    assert 'Gaussian-mixture' in r['fingerprint']['attention_scores']


def test_nested_modules_follow_all_runtime_arguments():
    helper='''class Inner(nn.Module):
 def forward(self,x,y):
  return F.linear(y,self.weight)
class Outer(nn.Module):
 def __init__(self):
  self.inner=Inner()
'''
    b=source('return self.outer.inner(x,x)',extra=helper,ctor='self.outer=Outer()')
    a=source('return self.outer.inner(x,x*x)',extra=helper,ctor='self.outer=Outer()')
    assert review_pair(b,b,'openevolve_v21')['classification']=='preserving'
    assert review_pair(b,a,'openevolve_v21') is None


def test_cached_additive_sine_features_preserve_position_mechanism():
    t=TRANSFORMER
    a=t.replace('self.position_embedding=nn.Embedding(20,6)',"self.register_buffer('position_code',torch.sin(torch.arange(120).reshape(20,6)))").replace('self.position_embedding(pos)','self.position_code[:idx.shape[1]]')
    r=review_pair({'model.py':t},{'model.py':a},'openevolve_v21')
    assert r['classification']=='preserving' and r['fingerprint_complete']
    assert 'additive' in r['fingerprint']['position']


def test_root_helper_token_decoder_is_followed_into_lookup():
    t=TRANSFORMER.replace('self.token_embedding=nn.Embedding(114,6)','self.token_embedding=nn.Embedding(114,3)')
    t=t.replace(' def forward(self,idx):',' def token_code(self):\n  z=self.token_embedding.weight\n  return torch.cat((z,z.square()),dim=-1)\n def forward(self,idx):')
    t=t.replace('hidden=self.token_embedding(idx)','code=self.token_code()\n  hidden=F.embedding(idx,code)')
    r=review_pair({'model.py':TRANSFORMER},{'model.py':t},'openevolve_v21')
    assert r['classification']=='changing' and r['fingerprint_complete']
    assert 'nonlinear decoded' in r['fingerprint']['embedding']


def test_left_coefficient_multiplication_is_not_final_feature_affine():
    b=source('return F.linear(x,self.output.weight)')
    a=source('return F.linear(self.mixing@x,self.output.weight)',ctor='self.mixing=nn.Parameter(torch.zeros(4,4))')
    assert review_pair(b,a,'openevolve_v21') is None
    fixed=source('return self.mixing@x',ctor="self.register_buffer('mixing',torch.ones(4,4))")
    assert review_pair(source('return x'),fixed,'openevolve_v21') is None


def test_independent_bias_and_constant_logit_coordinates_preserve():
    base={'model.py':TRANSFORMER}
    no_position={'model.py':TRANSFORMER.replace('+self.position_embedding(pos)','')}
    r=review_pair(base,no_position,'openevolve_v21')
    assert r['classification']=='preserving' and r['fingerprint_complete']
    t=TRANSFORMER.replace('return self.output(hidden)',"logits=self.output(hidden)\n  zero=logits.new_zeros(logits.shape[:-1]+(1,))\n  return torch.cat((zero,logits),dim=-1)")
    r=review_pair(base,{'model.py':t},'openevolve_v21')
    assert r['classification']=='preserving' and r['fingerprint_complete']
    wrong={'model.py':t.replace('dim=-1)','dim=1)')}
    r=review_pair(base,wrong,'openevolve_v21')
    assert r is None or r['classification']=='changing'


def test_final_logit_repetition_does_not_erase_history_repetition():
    t=TRANSFORMER.replace('return self.output(hidden)',"logits=self.output(hidden)\n  first=logits[..., :1]\n  return first.expand(*first.shape[:-1],114)")
    r=review_pair({'model.py':TRANSFORMER},{'model.py':t},'openevolve_v21')
    assert r is None or r['classification']=='preserving'
    wrong=t.replace('first.shape[:-1]','first.shape[1:]')
    assert review_pair({'model.py':t},{'model.py':wrong},'openevolve_v21') is None


def test_piecewise_token_table_storage_keeps_digit_decomposition_visible():
    t=TRANSFORMER.replace('self.token_embedding=nn.Embedding(114,6)', 'self.pair_embedding=nn.Embedding(100,5)\n  self.other_embedding=nn.Embedding(114,6)\n  self.token_projection=nn.Linear(6,6)')
    t=t.replace('hidden=self.token_embedding(idx)+self.position_embedding(pos)', '''pair=(idx-10).clamp(0,99)
  features=torch.where(((idx>=10)&(idx<110)).unsqueeze(-1),F.pad(self.pair_embedding(pair),(0,1)),self.other_embedding(idx))
  hidden=self.token_projection(features)+self.position_embedding(pos)''')
    r=review_pair({'model.py':TRANSFORMER},{'model.py':t},'openevolve_v21')
    assert r['classification']=='preserving' and r['fingerprint_complete']
    divided={'model.py':t.replace('self.pair_embedding(pair)','self.pair_embedding(pair//10)')}
    r=review_pair({'model.py':t},divided,'openevolve_v21')
    assert r is None or r['classification']=='changing'


def test_inline_nonlinear_table_coordinates_are_bound_to_lookup_coefficients():
    t=TRANSFORMER.replace('hidden=self.token_embedding(idx)+self.position_embedding(pos)', '''z=self.token_embedding.weight
  extra=z.square().sum(dim=-1,keepdim=True)
  table=torch.cat((z,extra),dim=-1)
  hidden=F.embedding(idx,table)+self.position_embedding(pos)''')
    r=review_pair({'model.py':TRANSFORMER},{'model.py':t},'openevolve_v21')
    assert r['classification']=='changing' and r['fingerprint_complete']
    assert 'nonlinear decoded' in r['fingerprint']['embedding']


def test_tail_tying_in_lag_table_is_not_clipping_the_scores():
    t=TRANSFORMER.replace('scores=q@k.transpose(-1,-2)', '''positions=torch.arange(x.shape[1])
  lag=(positions[:,None]-positions[None,:]).clamp_min(0)
  scores=self.bias[lag]''')
    b={'model.py':t};a={'model.py':t.replace('self.bias[lag]','self.bias[lag.clamp_max(8)]')}
    assert review_pair(b,a,'openevolve_v21')['classification']=='preserving'
    wrong={'model.py':t.replace('self.bias[lag]','self.bias[lag].clamp_max(8)')}
    r=review_pair(b,wrong,'openevolve_v21')
    assert r is None or r['classification']=='changing'


def test_constructor_nested_replacement_and_sequence_alias_are_followed():
    helper='''class Inner(nn.Module):
 def __init__(self):
  self.proj=nn.Linear(4,4)
 def forward(self,x):
  return self.proj(x)
'''
    b=source('return self.inner(x)',extra=helper,ctor='self.inner=Inner()')
    a=source('return self.inner(x)',extra=helper,ctor='self.inner=Inner()\n  self.inner.proj=nn.ReLU()')
    assert review_pair(b,a,'openevolve_v21') is None
    alias=source('return self.output_projection(x)',ctor='self.layers=nn.Sequential(nn.Linear(4,4),nn.ReLU())\n  projection=self.layers[0]\n  self.output_projection=projection')
    assert review_pair(source(),alias,'openevolve_v21')['classification']=='preserving'
    nonlinear={'model.py':alias['model.py'].replace('projection=self.layers[0]','projection=self.layers[1]')}
    assert review_pair(alias,nonlinear,'openevolve_v21') is None
    unknown=source('return self.inner(x)',extra=helper,ctor='self.inner=Inner()\n  self.inner.proj=unknown_factory()')
    assert review_pair(b,unknown,'openevolve_v21') is None
    two=source('return self.left(x)+self.right(x)',extra=helper,ctor='self.left=Inner()\n  self.right=Inner()\n  self.left.proj=nn.ReLU()')
    assert 'instance-specific' in profile(two)['error']


def test_documented_feature_normalization_coordinates_preserve():
    b={'model.py':TRANSFORMER}
    a={'model.py':TRANSFORMER.replace('return self.output(hidden)','normalized=F.layer_norm(hidden,(6,))\n  normalized=normalized-normalized.mean(dim=-1,keepdim=True)\n  return self.output(normalized)')}
    r=review_pair(b,a,'openevolve_v21')
    assert r['classification']=='preserving' and r['fingerprint_complete']
    bad={'model.py':a['model.py'].replace('mean(dim=-1','mean(dim=1')}
    assert review_pair(b,bad,'openevolve_v21') is None


def test_learned_scalar_factors_on_relative_tables_preserve():
    t=TRANSFORMER.replace('self.qkv=nn.Linear(6,18)', 'self.qkv=nn.Linear(6,18)\n  self.bias=nn.Parameter(torch.zeros(20,1))\n  self.gain=nn.Parameter(torch.ones(1,2))').replace('scores=q@k.transpose(-1,-2)', '''positions=torch.arange(x.shape[1])
  lag=(positions[:,None]-positions[None,:]).clamp_min(0)
  table=self.bias
  scores=q@k.transpose(-1,-2)+table[lag].permute(2,0,1)''')
    b={'model.py':t};a={'model.py':t.replace('table=self.bias','table=self.bias*self.gain')}
    r=review_pair(b,a,'openevolve_v21')
    assert r['classification']=='preserving' and r['fingerprint_complete']


def test_tuple_projection_outputs_keep_each_tensor_identity():
    helper='''class Projections(nn.Module):
 def __init__(self):
  self.q=nn.Linear(6,6)
  self.k=nn.Linear(6,6)
  self.v=nn.Linear(6,6)
 def forward(self,x):
  return self.q(x),self.k(x),self.v(x)
'''
    t=TRANSFORMER.replace('class Attention',helper+'class Attention').replace('self.qkv=nn.Linear(6,18)','self.qkv=Projections()').replace('self.qkv(x).chunk(3,dim=-1)','self.qkv(x)')
    r=review_pair({'model.py':TRANSFORMER},{'model.py':t},'openevolve_v21')
    assert r['classification']=='preserving' and r['fingerprint_complete']
    wrong={'model.py':t.replace('self.k(x),self.v(x)','self.k(x*x),self.v(x)')}
    assert review_pair({'model.py':t},wrong,'openevolve_v21') is None


def test_finite_adjudication_requires_both_complete_source_hashes(monkeypatch):
    import hashlib,json
    import experiments.ontology_review_addition as review
    before=source();after=source('return self.output(x)+self.bias',ctor='self.bias=nn.Parameter(torch.zeros(4))')
    digest=lambda s:hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    proof={'classification':'preserving','fingerprint':{},'evidence':[{'kind':'unit fixture binding proof'}]}
    monkeypatch.setattr(review,'pair_references',lambda:{('openevolve_v21',digest(before),digest(after)):proof})
    assert review.exact_pair_review(before,after,'openevolve_v21')['classification']=='preserving'
    for old,new in [({**before,'train.py':'changed training context'},after),(before,{k:v+'\n# changed source' for k,v in after.items()})]:
        assert review.exact_pair_review(old,new,'openevolve_v21') is None
    assert review.exact_pair_review(before,after,'tiny_adderboard_v21') is None


def test_ellipsis_with_two_slices_is_not_a_feature_only_map():
    b=source('return self.output(x)')
    a=source('return self.output(x[..., :1, :])')
    assert review_pair(b,a,'openevolve_v21') is None
    assert review_pair(b,source('return self.output(x[..., :1])'),'openevolve_v21')['classification']=='preserving'


def test_reviewed_invalid_constructor_has_source_and_campaign_witnesses():
    from experiments.ontology_review_addition import pair_references
    invalid=[r for r in pair_references().values() if r['classification']=='invalid_source']
    assert invalid
    for r in invalid:
        assert r['invalid_kind']=='undefined_constructor'
        assert r['implemented'] is False and r['executable'] is False
        assert r['missing_symbols'] and r['evidence']
        assert any(e['kind']=='undefined reached constructor' for e in r['evidence'])
        assert any(e['kind']=='independent campaign execution failure' and e['evaluation']['valid'] is False and e['evaluation']['failure_kind']=='execution' for e in r['evidence'])


def test_constructor_condition_uses_the_actual_instance_argument():
    helper='''class Wrapper(nn.Module):
 def __init__(self,nonlinear=False):
  self.inner=nn.ReLU() if nonlinear else nn.Linear(4,4)
 def forward(self,x):
  return self.inner(x)
'''
    b=source('return self.wrapper(x)',extra=helper,ctor='self.wrapper=Wrapper(False)')
    a=source('return self.wrapper(x)',extra=helper,ctor='self.wrapper=Wrapper(True)')
    assert review_pair(source(),b,'openevolve_v21')['classification']=='preserving'
    assert review_pair(b,a,'openevolve_v21') is None
    unknown=source('return self.wrapper(x)',extra=helper,ctor='self.wrapper=Wrapper(runtime_choice())')
    assert review_pair(b,unknown,'openevolve_v21') is None


def test_clipping_categorical_ids_is_token_row_tying_not_digit_encoding():
    b={'model.py':TRANSFORMER}
    a={'model.py':TRANSFORMER.replace('self.token_embedding(idx)','self.token_embedding(idx.clamp(0,12))')}
    r=review_pair(b,a,'openevolve_v21')
    assert r['classification']=='preserving' and r['fingerprint_complete']
    divided={'model.py':TRANSFORMER.replace('self.token_embedding(idx)','self.token_embedding(idx//10)')}
    r=review_pair(b,divided,'openevolve_v21')
    assert r is None or r['classification']=='changing'


def test_normalization_of_token_ids_is_not_hidden_feature_normalization():
    b={'model.py':TRANSFORMER}
    a={'model.py':TRANSFORMER.replace('self.token_embedding(idx)','self.token_embedding(F.layer_norm(idx,(20,)))')}
    assert review_pair(b,a,'openevolve_v21') is None


TARGET_BASE='''
def encode_targets(answer_digits: torch.Tensor) -> torch.Tensor:
 return answer_digits
def decode_targets(generated: torch.Tensor) -> torch.Tensor:
 return generated[:, :5]
'''
TARGET_BASE4='''
def encode_targets(answer_digits: torch.Tensor) -> torch.Tensor:
 residues = answer_digits.remainder(4)
 quotients = torch.div(answer_digits, 4, rounding_mode='floor')
 return torch.cat((residues, quotients), dim=1)
def decode_targets(generated: torch.Tensor) -> torch.Tensor:
 return generated[:, :5] + 4 * generated[:, 5:10]
'''


def test_output_digit_factorization_is_a_complete_changing_witness():
    r=review_pair({'model.py':TRANSFORMER+TARGET_BASE},{'model.py':TRANSFORMER+TARGET_BASE4},'openevolve_v21')
    assert r['classification']=='changing' and r['fingerprint_complete']
    assert 'output_factorization' in r['changed_components']
    assert 'residue' in r['fingerprint']['output_factorization']
    assert 'ten autoregressive' in r['fingerprint']['iteration']


def test_constant_interleaved_markers_preserve_with_explicit_iteration_detail():
    markers='''
def encode_targets(answer_digits: torch.Tensor) -> torch.Tensor:
 markers = torch.full_like(answer_digits[:, :4], 10)
 interleaved = torch.stack((answer_digits[:, :4], markers), dim=-1).flatten(1)
 return torch.cat((interleaved, answer_digits[:, 4:5]), dim=1)
def decode_targets(generated: torch.Tensor) -> torch.Tensor:
 return generated[:, ::2][:, :5]
'''
    r=review_pair({'model.py':TRANSFORMER+TARGET_BASE},{'model.py':TRANSFORMER+markers},'openevolve_v21')
    assert r['classification']=='preserving' and r['fingerprint_complete']
    assert 'nine autoregressive' in r['fingerprint']['iteration']
    assert 'discard marker' in r['fingerprint']['output_factorization']
    unknown=markers.replace('generated[:, ::2][:, :5]','generated[:, ::2][:, :5] + generated[:, 1:2]')
    r=review_pair({'model.py':TRANSFORMER+TARGET_BASE},{'model.py':TRANSFORMER+unknown},'openevolve_v21')
    assert r is None or not r['fingerprint_complete']


def test_column_specific_decimal_alphabet_is_preserving_and_explicit():
    phases='''
def encode_targets(answer_digits: torch.Tensor) -> torch.Tensor:
 phases = torch.arange(answer_digits.shape[1], device=answer_digits.device).unsqueeze(0)
 return 112 + 10 * phases + answer_digits
def decode_targets(generated: torch.Tensor) -> torch.Tensor:
 return (generated[:, :5] - 112) % 10
'''
    r=review_pair({'model.py':TRANSFORMER+TARGET_BASE},{'model.py':TRANSFORMER+phases},'openevolve_v21')
    assert r['classification']=='preserving' and r['fingerprint_complete']
    assert 'column' in r['fingerprint']['output_factorization']


def test_target_codec_reference_does_not_survive_constant_or_global_changes():
    from experiments.ontology_review_addition import target_descriptor
    import ast
    method=next(n for n in ast.parse(TARGET_BASE4).body if n.name=='encode_targets')
    assert target_descriptor(method,{'model.py':TRANSFORMER+TARGET_BASE4})
    assert target_descriptor(method,{'model.py':TRANSFORMER+TARGET_BASE4+'\ntorch=fake_library\n'}) is None
    bad=TARGET_BASE4.replace('remainder(4)','remainder(3)')
    r=review_pair({'model.py':TRANSFORMER+TARGET_BASE},{'model.py':TRANSFORMER+bad},'openevolve_v21')
    assert r is None or not r['fingerprint_complete']


def test_scaled_affine_head_replication_preserves_only_with_other_axes_fixed():
    b=source(ctor='self.n_head=2')
    body='h=self.output(x)*0.1\nreturn h.unsqueeze(2).expand(-1,-1,self.n_head,-1)'
    a=source(body,ctor='self.n_head=2')
    assert review_pair(b,a,'openevolve_v21')['classification']=='preserving'
    for bad in [body.replace('expand(-1,-1,','expand(-1,3,'),body.replace('unsqueeze(2)','unsqueeze(1)')]:
        assert review_pair(b,source(bad,ctor='self.n_head=2'),'openevolve_v21') is None


def test_indexed_learned_coefficients_keep_the_complete_nonlinear_path():
    for construction in ['nn.ModuleList([nn.Linear(6,6)])','nn.ModuleList([nn.Linear(6,6) for _ in range(2)])']:
        t=TRANSFORMER.replace('self.output=nn.Linear(6,114)','self.output=nn.Linear(6,114)\n  self.projections='+construction)
        before={'model.py':t.replace('return self.output(hidden)','return self.output(F.linear(hidden,torch.sin(self.projections[0].weight)))')}
        after={'model.py':before['model.py'].replace('torch.sin(self.projections[0].weight)','torch.sin(self.projections[0].weight @ self.projections[0].weight)')}
        assert profile(before)['parameter_nonlinear_sites']
        assert profile(after)['parameter_nonlinear_sites']!=profile(before)['parameter_nonlinear_sites']
        assert review_pair(before,after,'openevolve_v21') is None
        assert review_pair(before,before,'openevolve_v21')['fingerprint_complete']


def test_indexed_custom_parameter_properties_are_interpreted():
    helper='''class Kernel(nn.Module):
 def __init__(self):
  self.raw=nn.Parameter(torch.zeros(6,6))
 @property
 def weight(self):
  return torch.sin(self.raw)
'''
    t=TRANSFORMER.replace('class Attention',helper+'class Attention').replace('self.output=nn.Linear(6,114)','self.output=nn.Linear(6,114)\n  self.projections=nn.ModuleList([Kernel()])').replace('return self.output(hidden)','return self.output(F.linear(hidden,self.projections[0].weight))')
    before={'model.py':t};after={'model.py':t.replace('torch.sin(self.raw)','torch.sin(self.raw @ self.raw)')}
    assert any(s['scope']=='Kernel.weight' for s in profile(before)['parameter_nonlinear_sites'])
    assert review_pair(before,after,'openevolve_v21') is None


def test_unresolved_heterogeneous_module_attributes_do_not_become_fixed():
    t=TRANSFORMER.replace('self.output=nn.Linear(6,114)','self.output=nn.Linear(6,114)\n  self.projections=nn.ModuleList([nn.Linear(6,i) for i in widths])').replace('return self.output(hidden)','return self.output(F.linear(hidden,torch.sin(self.projections[0].weight)))')
    assert review_pair({'model.py':t},{'model.py':t},'openevolve_v21') is None


def test_p25_positional_shift_reference_is_complete_and_source_bound():
    from experiments.ontology_review_addition import pair_references
    key=(
        'openevolve_v21',
        '8eaa0daee166a96e219bce3441ee8b1b18167defb08d297b0dcd3df7abbc1fea',
        '558cf96c3e122309e7d0832d1266bfa4acafce6607d7ccd299da7e61ea087611',
    )
    result=pair_references()[key]
    assert result['classification']=='preserving'
    assert result['fingerprint_complete'] is True
    assert {'embedding','position','parameter_construction','sharing','output'}<=set(result['changed_components'])
    assert any(e['kind']=='directly source-adjudicated coordinate compensation' for e in result['evidence'])


def test_b04_c3_p25_energy_ratio_reference_is_separate_and_complete():
    from experiments.ontology_review_addition import pair_references
    key=(
        'openevolve_v21',
        '583ca995d2a46bb294e023312129c6399e2eb097bb426001ec06b7d0f6af1f01',
        '157cb16173b94b57643d691d2ea7eadd42b5851ce8b0487ae863560b4caec79f',
    )
    result=pair_references()[key]
    assert result['classification']=='preserving'
    assert result['fingerprint_complete'] is True
    assert {'position','parameter_construction','projection_relations'}<=set(result['changed_components'])
    assert 'fourth-root' in result['fingerprint']['position']
    assert 'query bias' in result['fingerprint']['projection_relations']


def test_b01_c3_p78_quadratic_bias_generator_is_changing_and_complete():
    from experiments.ontology_review_addition import pair_references
    key=(
        'openevolve_v21',
        'd196903d89fd3e8d628d836fa16a42c646296d1ca247cdbf55ccd85c86d140e0',
        '9d3d5f4c64266fe6b1f39f16865ecd275517790d12a3a0fd8b705fc092082424',
    )
    result=pair_references()[key]
    assert result['classification']=='changing'
    assert result['fingerprint_complete'] is True
    assert 'square' in result['fingerprint']['parameter_construction']
    assert result['family_signature']['parameter_generator']==[['learned nonlinear coefficient generator',['square']]]


def test_b01_c3_p95_head_axis_projection_simplification_is_bound():
    from experiments.ontology_review_addition import pair_references
    key=(
        'openevolve_v21',
        '271a44874d2591610ee5feb4fb1e12e872597058c13a143e86a183f16a04e120',
        '02a00311a4917693a933e2468283168b1dabbad263b8fe215970c2d37e831e54',
    )
    result=pair_references()[key]
    assert result['classification']=='preserving'
    assert result['fingerprint_complete'] is True
    assert 'head axis' in result['fingerprint']['sharing']
    assert 'first hidden feature' in result['fingerprint']['projection_relations']
    assert result['family_signature']['relative_scores'] is False


def test_b01_c3_p96_relative_bucket_boundary_is_changing():
    from experiments.ontology_review_addition import pair_references
    key=(
        'openevolve_v21',
        '6970c95f1d5964401bbf0fe470a1e8500442e26bd1dffcac274ac84f595c8183',
        '3b9e5c50db7591e4a46e9e026d09ae09e0276575d7c0f977a33791dbe2dcaee8',
    )
    result=pair_references()[key]
    assert result['classification']=='changing'
    assert result['fingerprint_complete'] is True
    assert 'distances 11' in result['fingerprint']['routing']
    assert result['family_signature']['relative_scores'] is True
    assert 'terminal bucket' in result['family_signature']['score_bias']
    parent=result['parent_family_signature']
    assert parent['position'].endswith('distance 14')
    assert parent['score_bias'].endswith('starts at 14')
    assert parent['position'] != result['family_signature']['position']
    assert parent['score_bias'] != result['family_signature']['score_bias']


def test_b01_c3_p98_head_sum_and_fixed_basis_are_changing():
    from experiments.ontology_review_addition import pair_references
    key=(
        'openevolve_v21',
        '02a00311a4917693a933e2468283168b1dabbad263b8fe215970c2d37e831e54',
        '438f59181110b058748a05ad313ea63eb234bc246f0035f06143761ed4af66f1',
    )
    result=pair_references()[key]
    assert result['classification']=='changing'
    assert result['fingerprint_complete'] is True
    assert 'summed' in result['fingerprint']['aggregation']
    assert 'fixed mean-zero' in result['fingerprint']['output']
    assert 'bias-free' in result['fingerprint']['feedforward']


def test_b01_c3_p99_distinct_head_sum_and_mlp_bias_change_is_bound():
    from experiments.ontology_review_addition import pair_references
    key=(
        'openevolve_v21',
        '5ff850b8a136fd71edc57cbda1e2b8501d998e51a4828e45bec16e1aebc1b467',
        '43802302ed0c23c2bfa31b208a84d9d9940d31d3434fe2633eca31d818f73bda',
    )
    result=pair_references()[key]
    assert result['classification']=='changing'
    assert result['fingerprint_complete'] is True
    assert 'tied-bias fc2' in result['fingerprint']['feedforward']
    assert 'selected feature' in result['fingerprint']['projection_relations']


def test_published_normalizer_conflicts_are_source_bound_and_do_not_share_families():
    """All eight live family conflicts retain their direct normalizer witness."""
    root=next(parent for parent in Path(__file__).resolve().parents if (parent/'data'/'c0c3').is_dir())
    campaign=root/'data'/'c0c3'/'controlled-openevolve-transformer-v2-1-mps-campaign'/'runs'
    cases=[
        ('b01-c1','d028ad47218b86e28f80c83cd2a8082552ecabd5fbcd4f55c2d7594d31aebbb6','3a620bf0ff41f8ad24a4d44596f333033150a7316c6967dfba1f59d17c6a0dca',(('QuotientFinalLayerNorm','quotient'),('TwoPrunedBiasLayerNorm','partial-coordinate'))),
        ('b02-c4','770094b8c9c85c80d1448f9bf2684c0557e209b9eaf59fdf8ecccf61992bbdb1','06be4144efb2c6fae0fb6a8e0a9f36a2ec741580a523136402f58a63db647dba',(('PartialAttentionLayerNorm','partial-coordinate'),('PartialScaleLayerNorm','partial-coordinate'))),
        ('b02-c4','658ba09a71e07c645a442a0ca66e2ea4e700d188a10a7285d3ed54579490a344','957b260d7057b069cd0824da039b71f15968650a5db7607bd34eb53990221140',(('PartialAttentionLayerNorm','partial-coordinate'),('PartialScaleLayerNorm','partial-coordinate'))),
        ('b04-c3','f7a258940967f45248222be17fd7877555b06933d44e43cd9412e67d7321ce5d','1e15722dfc38a8dabdbb6904fc85ba39103f62f7b0d776fc3f1f247ef6050694',(('RMSNorm','rms'),)),
        ('b04-c3','5e5346c916192e5938a0af89e1a4e4443e86e74375d72bd13aac13b772504982','2a402598c98e1b775e512672a8ac4bf1d17e0dbc192dbec62c4be09bde74ff51',(('RMSNorm','rms'),)),
        ('b04-c3','2b9cea175ed60b009337ac54f6e8ad23bf2c014eb4080237777bfb70bfd9fd5b','fcfc83eba5520cc2f55b313205844eb1bd2b00085993df48b45720d906cb7f58',(('RMSNorm','rms'),)),
        ('b04-c3','2b9cea175ed60b009337ac54f6e8ad23bf2c014eb4080237777bfb70bfd9fd5b','1b4be6decd2d36f4476f3f90b8db9757996d4279cf9b19ec5eb0b0e55e9fcb7b',(('RMSNorm','rms'),)),
        ('b04-c3','2b9cea175ed60b009337ac54f6e8ad23bf2c014eb4080237777bfb70bfd9fd5b','69370d9405932df51c00f4bd20583f2abec37eeb2ddae471309f41a2a27655aa',(('RMSNorm','rms'),)),
    ]
    prefix='controlled-openevolve-transformer-v2-1-source-only-ten-digit-addition-pair-transformer-openevolve-v2-1-mps-openevolve-'
    def sources(run,candidate):
        folder=Path('\\\\?\\'+str(campaign/(prefix+run)/'candidates'/candidate))
        return {str(path.relative_to(folder)).replace('\\','/'):path.read_text(encoding='utf-8') for path in folder.rglob('*.py')}
    for run,parent,child,topology in cases:
        result=review_pair(sources(run,parent),sources(run,child),'openevolve_v21')
        assert result['classification']=='changing'
        assert result['fingerprint_complete'] is True
        assert 'normalization' in result['changed_components']
        assert tuple(map(tuple,result['family_signature']['normalization_topology']))==topology


def test_exact_pair_normalization_refresh_separates_legacy_rms_bridges():
    """Fresh source topology may supplement, but never replace, direct evidence."""
    import copy
    from experiments.ontology_review_addition import pair_references, task_fingerprint
    from experiments.ontology_review_service import reconcile_families

    root=next(parent for parent in Path(__file__).resolve().parents if (parent/'data'/'c0c3').is_dir())
    campaign=root/'data'/'c0c3'/'controlled-openevolve-transformer-v2-1-mps-campaign'/'runs'
    prefix='controlled-openevolve-transformer-v2-1-source-only-ten-digit-addition-pair-transformer-openevolve-v2-1-mps-openevolve-b04-c3'
    def sources(candidate):
        folder=Path('\\\\?\\' + str(campaign/prefix/'candidates'/candidate))
        return {str(path.relative_to(folder)).replace('\\','/'):path.read_text(encoding='utf-8') for path in folder.rglob('*.py')}
    def family(source):
        _,complete,value=task_fingerprint(source,profile(source),'openevolve_v21')
        assert complete and value
        return value

    changing=[
        ('f7a258940967f45248222be17fd7877555b06933d44e43cd9412e67d7321ce5d','1e15722dfc38a8dabdbb6904fc85ba39103f62f7b0d776fc3f1f247ef6050694'),
        ('5e5346c916192e5938a0af89e1a4e4443e86e74375d72bd13aac13b772504982','2a402598c98e1b775e512672a8ac4bf1d17e0dbc192dbec62c4be09bde74ff51'),
        ('2b9cea175ed60b009337ac54f6e8ad23bf2c014eb4080237777bfb70bfd9fd5b','fcfc83eba5520cc2f55b313205844eb1bd2b00085993df48b45720d906cb7f58'),
        ('2b9cea175ed60b009337ac54f6e8ad23bf2c014eb4080237777bfb70bfd9fd5b','1b4be6decd2d36f4476f3f90b8db9757996d4279cf9b19ec5eb0b0e55e9fcb7b'),
        ('2b9cea175ed60b009337ac54f6e8ad23bf2c014eb4080237777bfb70bfd9fd5b','69370d9405932df51c00f4bd20583f2abec37eeb2ddae471309f41a2a27655aa'),
    ]
    changing_rows=[]
    for parent,child in changing:
        result=review_pair(sources(parent),sources(child),'openevolve_v21')
        assert result['classification']=='changing'
        assert 'normalization' in result['changed_components']
        assert result['family_signature']['normalization_modules']==('RMSNorm',)
        assert tuple(map(tuple,result['family_signature']['normalization_topology']))==(('RMSNorm','rms'),)
        changing_rows.append((parent,child,result))

    p25_parent='a8d37b0d9f299a3540c96f8c8f9f68a78c5ad8d1ddc6fc945d62226fa9456b80'
    p25_child='32361ba4ad394ef6bc7824bd8387606dd0f4007d89149538fd310a264a19b3ab'
    static=copy.deepcopy(pair_references()[('openevolve_v21','583ca995d2a46bb294e023312129c6399e2eb097bb426001ec06b7d0f6af1f01','157cb16173b94b57643d691d2ea7eadd42b5851ce8b0487ae863560b4caec79f')])
    p25=review_pair(sources(p25_parent),sources(p25_child),'openevolve_v21')
    for field in ('classification','fingerprint','notes','changed_components'):
        assert p25[field]==static[field]
    assert p25['evidence'][:-1]==static['evidence']
    assert p25['evidence'][-1]['kind']=='SHA-256-bound direct review'
    assert p25['family_signature']['normalization_modules']==('RMSNorm',)
    assert tuple(map(tuple,p25['family_signature']['normalization_topology']))==(('RMSNorm','rms'),)

    p30_parent='0c164423644fc4ed132e3dd035da35775cbe6d0151cf0f171d58207492bff64b'
    p30_child='7a831613849fe85c0b2bbe05599ad15029cb7cbe77cf1644fd32d96bc0957ba1'
    p30_before,p30_after=family(sources(p30_parent)),family(sources(p30_child))
    assert p30_before['normalization_modules']==p30_after['normalization_modules']==('RMSNorm',)
    assert tuple(map(tuple,p30_after['normalization_topology']))==(('RMSNorm','rms'),)

    baseline='2b9cea175ed60b009337ac54f6e8ad23bf2c014eb4080237777bfb70bfd9fd5b'
    p1_child='352e3e908ec8267a38b8df8029df98976763b82b058328a12f3f55e267a1724c'
    base_family,p1_family=family(sources(baseline)),family(sources(p1_child))
    assert base_family['normalization_modules']==p1_family['normalization_modules']==('LayerNorm',)
    rows=[
        {'source_sha256':baseline,'fingerprint_complete':True,'family_signature':base_family,'classification':'preserving','parent_reviews':[]},
        {'source_sha256':p1_child,'fingerprint_complete':True,'family_signature':p1_family,'classification':'preserving','parent_reviews':[{'source_sha256':baseline,'classification':'preserving'}]},
        {'source_sha256':p25_parent,'fingerprint_complete':True,'family_signature':family(sources(p25_parent)),'classification':'preserving','parent_reviews':[]},
        {'source_sha256':p25_child,'fingerprint_complete':True,'family_signature':p25['family_signature'],'classification':'preserving','parent_reviews':[{'source_sha256':p25_parent,'classification':'preserving'}]},
        {'source_sha256':p30_parent,'fingerprint_complete':True,'family_signature':p30_before,'classification':'preserving','parent_reviews':[]},
        {'source_sha256':p30_child,'fingerprint_complete':True,'family_signature':p30_after,'classification':'preserving','parent_reviews':[{'source_sha256':p30_parent,'classification':'preserving'}]},
    ]
    rows.extend({'source_sha256':child,'fingerprint_complete':True,'family_signature':result['family_signature'],'classification':'changing','parent_reviews':[{'source_sha256':parent,'classification':'changing'}]} for parent,child,result in changing_rows)
    assert reconcile_families(rows)==[]
    assert all(row['classification']!='uncertain' for row in rows)


def test_b04_c3_v9_source_bound_cohort_has_complete_records():
    """Every V9 cohort verdict is tied to both archived source snapshots."""
    import json
    from experiments.ontology_addition_b04_references import references
    published = json.loads(Path('outputs/ontology/openevolve_v21.json').read_text(encoding='utf-8'))
    paths = {row['source_sha256']: row['source_path'] for row in published['rows']}

    def source_tree(root):
        root = Path(root)
        return {str(path.relative_to(root)).replace('\\', '/'): path.read_text(encoding='utf-8')
                for path in root.rglob('*.py')}

    results = []
    for (campaign, before, after), expected in references().items():
        actual = review_pair(source_tree(paths[before]), source_tree(paths[after]), campaign)
        assert actual['classification'] == expected['classification']
        assert actual['fingerprint_complete'] is True
        assert json.dumps(actual['family_signature'], sort_keys=True) == json.dumps(expected['family_signature'], sort_keys=True)
        assert any(item['kind'] == 'SHA-256-bound direct review' for item in actual['evidence'])
        results.append(actual['classification'])
    assert results.count('changing') == 4
    assert results.count('preserving') == 7


def test_b03_c2_v11_source_bound_gauge_cohort_is_complete_and_rejects_mutations():
    """Each V11 gauge verdict requires its complete archived source pair."""
    import json
    from experiments.ontology_addition_b03_references import references
    published = json.loads(Path('outputs/ontology/openevolve_v21.json').read_text(encoding='utf-8'))
    paths = {row['source_sha256']: row['source_path'] for row in published['rows']}

    def source_tree(root):
        root = Path(root)
        return {str(path.relative_to(root)).replace('\\', '/'): path.read_text(encoding='utf-8')
                for path in root.rglob('*.py')}

    results = []
    for (campaign, before, after), expected in references().items():
        parent, child = source_tree(paths[before]), source_tree(paths[after])
        actual = review_pair(parent, child, campaign)
        assert actual['classification'] == 'preserving'
        assert actual['fingerprint_complete'] is True
        assert json.dumps(actual['family_signature'], sort_keys=True) == json.dumps(expected['family_signature'], sort_keys=True)
        assert any(item['kind'] == 'SHA-256-bound direct review' for item in actual['evidence'])
        altered = {'src/model.py': child['src/model.py'] + '\n# V11 SHA mutation\n'}
        assert review_pair(parent, altered, campaign) is None
        results.append(actual)
    assert len(results) == 10
    assert all(result['changed_components'] == [] for result in results)


def test_b03_c1_v12_source_bound_mechanism_cohort_is_complete_and_rejects_mutations():
    """Every V12 verdict is bound to both full archived source snapshots."""
    import json
    from experiments.ontology_addition_b03c1_references import references
    published = json.loads(Path('outputs/ontology/openevolve_v21.json').read_text(encoding='utf-8'))
    paths = {row['source_sha256']: row['source_path'] for row in published['rows']}

    def source_tree(root):
        root = Path(root)
        return {str(path.relative_to(root)).replace('\\', '/'): path.read_text(encoding='utf-8')
                for path in root.rglob('*.py')}

    results = []
    for (campaign, before, after), expected in references().items():
        parent, child = source_tree(paths[before]), source_tree(paths[after])
        actual = review_pair(parent, child, campaign)
        assert actual['classification'] == 'changing'
        assert actual['fingerprint_complete'] is True
        assert actual['changed_components'] == expected['changed_components']
        assert json.dumps(actual['family_signature'], sort_keys=True) == json.dumps(expected['family_signature'], sort_keys=True)
        assert any(item['kind'] == 'SHA-256-bound direct review' for item in actual['evidence'])
        altered = {'src/model.py': child['src/model.py'] + '\n# V12 SHA mutation\n'}
        assert review_pair(parent, altered, campaign) is None
        results.append(actual)
    assert len(results) == 5
    assert all(result['classification'] == 'changing' for result in results)


def test_b01_c3_v13_source_bound_address_cohort_is_complete_and_rejects_mutations():
    """Every V13 transition requires the exact archived source pair."""
    import json
    from experiments.ontology_addition_b01c3_references import references
    published = json.loads(Path('outputs/ontology/openevolve_v21.json').read_text(encoding='utf-8'))
    paths = {row['source_sha256']: row['source_path'] for row in published['rows']}

    def source_tree(root):
        root = Path(root)
        return {str(path.relative_to(root)).replace('\\', '/'): path.read_text(encoding='utf-8')
                for path in root.rglob('*.py')}

    results = []
    for (campaign, before, after), expected in references().items():
        parent, child = source_tree(paths[before]), source_tree(paths[after])
        actual = review_pair(parent, child, campaign)
        assert actual['classification'] == 'changing'
        assert actual['fingerprint_complete'] is True
        assert actual['changed_components'] == expected['changed_components']
        assert json.dumps(actual['family_signature'], sort_keys=True) == json.dumps(expected['family_signature'], sort_keys=True)
        assert any(item['kind'] == 'SHA-256-bound direct review' for item in actual['evidence'])
        altered = {'src/model.py': child['src/model.py'] + '\n# V13 SHA mutation\n'}
        assert review_pair(parent, altered, campaign) is None
        results.append(actual)
    assert len(results) == 4
    assert all(result['classification'] == 'changing' for result in results)


def test_b05_c3_v14_source_bound_relative_cohort_is_complete_and_rejects_mutations():
    """Every V14 relative-score transition requires its full archived source pair."""
    import json
    from experiments.ontology_addition_b05c3_references import references
    published = json.loads(Path('outputs/ontology/openevolve_v21.json').read_text(encoding='utf-8'))
    paths = {row['source_sha256']: row['source_path'] for row in published['rows']}

    def source_tree(root):
        root = Path(root)
        return {str(path.relative_to(root)).replace('\\', '/'): path.read_text(encoding='utf-8')
                for path in root.rglob('*.py')}

    results = []
    for (campaign, before, after), expected in references().items():
        parent, child = source_tree(paths[before]), source_tree(paths[after])
        actual = review_pair(parent, child, campaign)
        assert actual['classification'] == 'changing'
        assert actual['fingerprint_complete'] is True
        assert actual['changed_components'] == expected['changed_components']
        expected_signature = dict(expected['family_signature'])
        for field in ('normalization_modules', 'normalization_topology'):
            expected_signature[field] = actual['family_signature'][field]
        assert json.dumps(actual['family_signature'], sort_keys=True) == json.dumps(expected_signature, sort_keys=True)
        assert any(item['kind'] == 'SHA-256-bound direct review' for item in actual['evidence'])
        altered = {'src/model.py': child['src/model.py'] + '\n# V14 SHA mutation\n'}
        assert review_pair(parent, altered, campaign) is None
        results.append(actual)
    assert len(results) == 4
    assert all(result['classification'] == 'changing' for result in results)


def test_v15_final_source_readable_residuals_are_complete_and_reject_mutations():
    """The final readable queue is exact-pair-bound; source absence is not hidden."""
    import json
    from experiments.ontology_addition_remaining_references import references
    published = json.loads(Path('outputs/ontology/openevolve_v21.json').read_text(encoding='utf-8'))
    paths = {row['source_sha256']: row['source_path'] for row in published['rows']}

    def source_tree(root):
        root = Path(root)
        return {str(path.relative_to(root)).replace('\\', '/'): path.read_text(encoding='utf-8')
                for path in root.rglob('*.py')}

    results = []
    for (campaign, before, after), expected in references().items():
        assert Path(paths[before]).is_dir() and Path(paths[after]).is_dir()
        parent, child = source_tree(paths[before]), source_tree(paths[after])
        actual = review_pair(parent, child, campaign)
        assert actual['classification'] == expected['classification']
        assert actual['fingerprint_complete'] is True
        assert actual['changed_components'] == expected['changed_components']
        expected_signature = dict(expected['family_signature'])
        for field in ('normalization_modules', 'normalization_topology'):
            expected_signature[field] = actual['family_signature'][field]
        assert json.dumps(actual['family_signature'], sort_keys=True) == json.dumps(expected_signature, sort_keys=True)
        assert any(item['kind'] == 'SHA-256-bound direct review' for item in actual['evidence'])
        altered = {'src/model.py': child['src/model.py'] + '\n# V15 SHA mutation\n'}
        assert review_pair(parent, altered, campaign) is None
        results.append(actual)
    assert len(results) == 10
    assert sum(result['classification'] == 'changing' for result in results) == 8
    assert sum(result['classification'] == 'preserving' for result in results) == 2
