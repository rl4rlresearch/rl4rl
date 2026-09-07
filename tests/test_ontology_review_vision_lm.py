import textwrap
import json
from pathlib import Path
from experiments.generate_fashion_candidate_catalog_v2 import WORKSPACE, source

from experiments.ontology_review_vision_lm import FASHION, NANO, _fixed_windows, profile, review_pair
import ast


def cnn(extra="", forward="return self.classifier(self.features(images))", groups=1):
    source = textwrap.dedent(f'''
        import torch
        from torch import nn
        import torch.nn.functional as F
        class ImageClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(nn.Conv2d(1, 8, 3, groups={groups}), nn.GELU(), nn.MaxPool2d(2))
                self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(8*13*13, 10))
    ''')
    source += textwrap.indent(extra, '        ') + '\n'
    source += '    def forward(self, images):\n' + textwrap.indent(textwrap.dedent(forward), '        ') + '\n'
    source += '\ndef build_model():\n    return ImageClassifier()\n'
    ast.parse(source)
    return {"train.py": source}


def test_standard_cnn_fingerprint_is_complete_and_width_preserving():
    a = cnn()
    b = {k:v.replace('8', '16') for k,v in a.items()}
    result = review_pair(a,b,FASHION)
    assert result['classification']=='preserving'
    assert result['fingerprint_complete']
    assert all(v is not None for v in result['fingerprint'].values())
    assert result['evidence']


def test_b03_c3_fixed_tta_numeric_cohort_has_exact_parent_source_bindings():
    catalog = json.loads((Path(__file__).parents[1] / 'experiments/ontology_fashion_candidate_references_v3_unresolved.json').read_text(encoding='utf8'))
    expected = {
        ('4ffbc0a1f767595e54b6c655ee0fbb2994855e5a1120b66d54ba3ceb5610dc86', '0bd0dad2b074bd38d30ec730821f3a82d7813f0c413d2fa7dc409a91960b7d53'),
        ('e0ad17733c7bf7999e9cd8f7688eaebd799cef045e3a9446b8a8746d74c41364', '50f75a7689b8372501e60728c85727a0e92ffa1e4b16fe44dc1b77c84b86fac3'),
        ('54f7c601df7b096e7061d56274c5a640045a7cae595d64df9b1337034057ba0e', '8813028f046d5eed618f169526d654aeb751e2587a764cff003b8ac979355498'),
        ('4ffbc0a1f767595e54b6c655ee0fbb2994855e5a1120b66d54ba3ceb5610dc86', 'f67074c9fe27e577d19d86f903e95d5a69c82fc9d64dc88ec83fdc41630a73d4'),
        ('e0ad17733c7bf7999e9cd8f7688eaebd799cef045e3a9446b8a8746d74c41364', '48820682bd3c4988f0393a9994f8ee13d679043f7a5c25d3284144ce7a29eefd'),
        ('e0ad17733c7bf7999e9cd8f7688eaebd799cef045e3a9446b8a8746d74c41364', 'a0339156db2288538cd44cbd8026369f3971499dcebf04f6c438e363525f43e1'),
        ('e0ad17733c7bf7999e9cd8f7688eaebd799cef045e3a9446b8a8746d74c41364', 'a9769fbb266c008f3596fd0b121f49c1714c6c59bf7eaf78109904ffb4d97823'),
    }
    rows = {(r['parent_source_sha256'], r['source_sha256']): r for r in catalog['reviewed_profiles']}
    assert expected <= rows.keys()
    for key in expected:
        result = rows[key]
        assert result['transition_classification'] == 'preserving'
        assert result['changed_components'] == []
        assert len(result['fingerprint']) == 27
        assert any(e['kind'] == 'directly source-adjudicated fixed translated/flip TTA mixture settings' for e in result['evidence'])


def test_new_summary_basis_changes_family():
    a=cnn();b={k:v.replace('MaxPool2d', 'AvgPool2d') for k,v in a.items()}
    result=review_pair(a,b,FASHION)
    assert result['classification']=='changing'
    assert result['changed_components']


def test_functional_flatten_keeps_batch_axis_and_rejects_partial_layouts():
    original = cnn(forward='return self.classifier(self.features(images).flatten(1))')
    functional = cnn(forward='return self.classifier(torch.flatten(self.features(images), 1))')
    assert review_pair(original, functional, FASHION)['classification'] == 'preserving'
    for expression in ['torch.flatten(self.features(images), 0)',
                       'torch.flatten(self.features(images), 1, 2)',
                       'self.features(images).flatten(1, end_dim=2)']:
        assert profile(cnn(forward='return self.classifier(' + expression + ')'), FASHION) is None


def test_conventional_feature_concatenation_is_recorded_without_hidden_routing():
    fixed=cnn(forward='return self.classifier(torch.cat((self.features(images), self.features(images)), dim=1))')
    result=profile(fixed,FASHION)
    assert result and 'concatenation' in result['fingerprint']['connectivity']
    moved={k:v.replace('dim=1','dim=0') for k,v in fixed.items()}
    assert profile(moved,FASHION) is None
    gated={k:v.replace('self.features(images), self.features(images)', 'self.features(images), images * images') for k,v in fixed.items()}
    assert profile(gated,FASHION) is None


def test_terminal_global_summary_changes_readout_even_with_existing_max_pool():
    plain=cnn()
    pooled=cnn(forward='features = self.features(images)\nsummary = F.adaptive_max_pool2d(features, 1).flatten(1)\nreturn self.classifier(summary)')
    result=review_pair(plain,pooled,FASHION)
    assert result['classification']=='changing'
    assert result['family_signature']['readout']=='global max feature descriptor'
    assert result['fingerprint']['aggregation']=='global max feature descriptor'


def test_unreviewed_feature_products_and_calls_are_not_silently_preserving():
    assert profile(cnn(forward='return self.classifier(self.features(images * images))'),FASHION) is None
    assert profile(cnn(forward='return self.classifier(self.features(1 / images))'),FASHION) is None
    assert profile(cnn(forward='return mystery(images)'),FASHION) is None
    assert profile(cnn(extra='self.gate = nn.Parameter(torch.zeros(8))'),FASHION) is None


def test_complete_candidate_profile_survives_pending_parent_review():
    result=review_pair(cnn(forward='return mystery(images)'),cnn(),FASHION)
    assert result['classification']=='uncertain'
    assert result['fingerprint_complete'] and result['evidence']
    assert review_pair({},cnn(),FASHION)['classification']=='uncertain'


def test_fixed_tta_alias_does_not_admit_confidence_weighted_routing():
    base=cnn()['train.py'].replace('def forward(self, images):','def _predict(self, images):')
    base=base[:base.index('def build_model')]
    wrapper='''
    def forward(self,images):
        if self.training:
            return self._predict(images)
        outputs=[]
        for view in (images,images.flip(-1)):
            outputs.append(self._predict(view))
        return torch.stack(outputs,dim=0).mean(dim=0)
'''
    fixed={'train.py':base+wrapper}
    assert profile(fixed,FASHION) is not None
    adaptive={'train.py':base+wrapper.replace('return torch.stack(outputs,dim=0).mean(dim=0)', 'weights = torch.softmax(torch.stack(outputs,dim=0),dim=0)\n        return (weights * torch.stack(outputs,dim=0)).sum(dim=0)')}
    assert profile(adaptive,FASHION) is None


def test_tta_rejects_value_dependent_control_coordinates_and_nonlinear_mixtures():
    base=cnn()['train.py'].replace('def forward(self, images):','def _predict(self, images):')
    base=base[:base.index('def build_model')]
    unsupported = [
        'if images.mean() > 0:\n    return self._predict(images)\nreturn self._predict(images.flip(-1))',
        'return self._predict(images) if images.mean() > 0 else self._predict(images.flip(-1))',
        'outputs=[self._predict(view) for view in (images,images.flip(-1)) if view.mean()>0]\nreturn torch.stack(outputs).mean(0)',
        'outputs=[]\nfor index in range(int(images.mean())):\n    outputs.append(self._predict(images))\nreturn torch.stack(outputs).mean(0)',
        'offset=int(images.mean())\nreturn self._predict(images[...,offset:])',
        'weight=1\nweight=images.mean()\nreturn weight*self._predict(images)',
        'return 1 / self._predict(images)',
        'return self._predict(images) ** 2',
        'return math.exp(images.mean()) * self._predict(images)',
        'self.training=images.mean()>0\nreturn self._predict(images)',
    ]
    for body in unsupported:
        source={'train.py':base+'\n    def forward(self,images):\n'+textwrap.indent(body,'        ')+'\n'}
        assert profile(source,FASHION) is None, body


def test_constructor_local_name_cannot_hide_a_feature_product():
    source=cnn()['train.py'].replace('super().__init__()', 'super().__init__()\n        channels = 8')
    source=source.replace('return self.classifier(self.features(images))', 'channels = images\n        return self.classifier(self.features(images * channels))')
    assert profile({'train.py':source},FASHION) is None


def test_standard_grammar_does_not_hide_state_tying_or_decorators():
    state=cnn(extra='self.counter = 1', forward='self.counter = images\nreturn self.classifier(self.features(images * self.counter))')
    assert profile(state,FASHION) is None
    tied=cnn(extra='self.classifier.weight = self.features.weight')
    assert profile(tied,FASHION) is None
    decorated={k:v.replace('def forward(self, images):','@custom_transform\n    def forward(self, images):') for k,v in cnn().items()}
    assert profile(decorated,FASHION) is None
    indexed=cnn(extra='self.history = [0]', forward='self.history[0] = images\nreturn self.classifier(self.features(images))')
    assert profile(indexed,FASHION) is None


def test_routine_activation_dropout_storage_and_module_aliases_keep_details():
    prelu={k:v.replace('nn.GELU()', 'nn.PReLU(num_parameters=8)') for k,v in cnn().items()}
    result=review_pair(cnn(),prelu,FASHION)
    assert result['classification']=='preserving'
    assert 'PReLU' in result['fingerprint']['activation']
    assert 'learned scalar/channel slopes' in result['fingerprint']['parameter_construction']
    dropout=cnn(forward='features = self.features(images)\nfeatures = F.dropout(features, p=0.15, training=self.training)\nreturn self.classifier(features)')
    assert 'F.dropout' in profile(dropout,FASHION)['fingerprint']['stochasticity']
    variable={k:v.replace('p=0.15','p=images') for k,v in dropout.items()}
    assert profile(variable,FASHION) is None
    layout=cnn(forward='images = images.contiguous(memory_format=torch.channels_last)\nreturn self.classifier(self.features(images))')
    assert review_pair(cnn(),layout,FASHION)['classification']=='preserving'
    assert 'storage_conversions' in profile(layout,FASHION)['settings']
    integer=cnn(forward='images = images.to(torch.int64)\nreturn self.classifier(self.features(images))')
    assert profile(integer,FASHION) is None
    aliased=cnn(forward='features = self.features\nx = features[0](images)\nx = features[1](x)\nx = features[2](x)\nreturn self.classifier(x)')
    assert profile(aliased,FASHION)
    rebound={k:v.replace('features = self.features','features = self.features\n        features = images') for k,v in aliased.items()}
    assert profile(rebound,FASHION) is None


def test_execution_overrides_and_method_rebinding_need_source_review():
    override=cnn()['train.py'].replace('\ndef build_model():', '\n    def __call__(self, images):\n        return images * images\n\ndef build_model():')
    assert profile({'train.py':override},FASHION) is None
    rebound=cnn(extra='self.forward = self.features')
    assert profile(rebound,FASHION) is None
    inherited={k:v.replace('class ImageClassifier(nn.Module)', 'class ImageClassifier(CustomBase)') for k,v in cnn().items()}
    assert profile(inherited,FASHION) is None


def _inference_source(index):
    refs=json.loads((Path(__file__).parents[1]/'experiments/ontology_fashion_inference_references.json').read_text(encoding='utf8'))['reviewed_wrappers']
    ref=refs[index]
    if ref.get('composition_only'):
        custom=json.loads((Path(__file__).parents[1]/'experiments/ontology_fashion_custom_references.json').read_text(encoding='utf8'))['reviewed_profiles'][ref['custom_reference_indices'][0]]
        return {'train.py':'\n\n'.join(custom['program_parts'].values())}
    base=cnn(extra='self.context_classifier = nn.Linear(8, 10)' if ref.get('additional_audit_index') == 8 else '')['train.py']
    base=base[:base.index('def build_model')]
    if ref['model_alias']:
        base=base.replace('def forward(self, images):', 'def '+ref['model_alias']+'(self, images):')
    else:
        base=base[:base.index('    def forward(')]
    return {'train.py':base+'\n'+ '\n\n'.join(textwrap.indent(source,'    ') for source in ref['source_methods'].values())+'\n'}


def test_every_audited_inference_body_replays_with_complete_fingerprint():
    refs=json.loads((Path(__file__).parents[1]/'experiments/ontology_fashion_inference_references.json').read_text(encoding='utf8'))['reviewed_wrappers']
    for i,ref in enumerate(refs):
        if ref.get('composition_only') and 'residual_wrapper_audit_index' in ref:
            # These are admission contracts for an independently reviewed
            # custom core. They cannot provide a standalone model fingerprint.
            from experiments.ontology_review_vision_lm import _inference_reference
            methods = {name:ast.parse(source).body[0] for name,source in ref['source_methods'].items()}
            assert _inference_reference(methods, ref['model_alias'])['composition_only'], i
            base = cnn()['train.py'].split('    def forward(')[0]
            unreviewed = {'train.py':base + '\n'.join(textwrap.indent(source, '    ') for source in ref['source_methods'].values())}
            assert profile(unreviewed, FASHION) is None, i
            continue
        result=profile(_inference_source(i),FASHION)
        assert result, i
        assert all(v is not None for v in result['fingerprint'].values()),i
        assert all(result['family_signature'].get(k)==v for k,v in ref['family_facts'].items()),i


def test_residual_power_means_preserve_but_cannot_admit_new_view_routing():
    refs=json.loads((Path(__file__).parents[1]/'experiments/ontology_fashion_inference_references.json').read_text(encoding='utf8'))['reviewed_wrappers']
    index=next(i for i,r in enumerate(refs) if r.get('residual_wrapper_audit_index')==0)
    fixed=_inference_source(index)
    result=review_pair(cnn(),fixed,FASHION)
    assert result['classification']=='preserving'
    assert 'power mean' in result['inference']['procedure']
    routed={name:source.replace('2.0 * weight * orientation_consensus.pow(fusion_power)', '2.0 * weight * orientation_consensus.max(dim=1, keepdim=True).values * orientation_consensus.pow(fusion_power)') for name,source in fixed.items()}
    assert routed!=fixed and profile(routed,FASHION) is None


def test_scalar_post_mixture_calibration_preserves_but_routing_and_votes_change():
    assert review_pair(_inference_source(112),_inference_source(113),FASHION)['classification']=='preserving'
    assert review_pair(_inference_source(48),_inference_source(51),FASHION)['classification']=='changing'
    hard=review_pair(_inference_source(28),_inference_source(29),FASHION)
    assert hard['classification']=='changing'
    assert 'hard' in hard['fingerprint']['aggregation']
    assert review_pair(_inference_source(43),_inference_source(46),FASHION)['classification']=='changing'


def test_inference_templates_preserve_dimensions_zero_switches_and_helpers():
    original=_inference_source(39)
    swept={k:v.replace('0.02 * torch.tanh','0.043 * torch.tanh') for k,v in original.items()}
    assert review_pair(original,swept,FASHION)['classification']=='preserving'
    for old,new in [('0.02 * torch.tanh','0.0 * torch.tanh'),('torch.tanh','torch.sin'),('topk(2,','topk(3,')]:
        changed={k:v.replace(old,new) for k,v in original.items()}
        assert changed!=original
        assert profile(changed,FASHION) is None
    shaped=_inference_source(41)
    reshaped={k:v.replace('reshape(len(views), images.shape[0], 10)','reshape(images.shape[0], len(views), 10)') for k,v in shaped.items()}
    assert shaped!=reshaped and profile(reshaped,FASHION) is None
    helper=_inference_source(80)
    altered={k:v.replace('return 0.5 * (self._forward_once(images) + self._forward_once(images.flip(-1)))','return self._forward_once(images) * self._forward_once(images.flip(-1))') for k,v in helper.items()}
    assert altered!=helper and profile(altered,FASHION) is None


def _additional_inference_source(audit_index):
    refs=json.loads((Path(__file__).parents[1]/'experiments/ontology_fashion_inference_references.json').read_text(encoding='utf8'))['reviewed_wrappers']
    index=next(i for i,ref in enumerate(refs) if ref.get('additional_audit_index') == audit_index)
    return _inference_source(index)


def test_inline_inference_is_fully_bound_and_records_new_summary_bases():
    fixed=_additional_inference_source(4)
    assert review_pair(cnn(),fixed,FASHION)['classification']=='preserving'
    minimum=review_pair(fixed,_additional_inference_source(32),FASHION)
    assert minimum['classification']=='changing'
    assert 'spatial minimum' in minimum['fingerprint']['input_transform']
    routed=review_pair(fixed,_additional_inference_source(36),FASHION)
    assert routed['classification']=='changing'
    assert 'class-label agreement' in routed['fingerprint']['routing']
    correction=review_pair(_additional_inference_source(45),_additional_inference_source(46),FASHION)
    assert correction['classification']=='changing'
    assert 'scatter-added' in correction['fingerprint']['output']
    conditional=profile(_additional_inference_source(55),FASHION)
    assert 'top-two class margin' in conditional['fingerprint']['conditional_compute']
    context=profile(_additional_inference_source(8),FASHION)
    assert context['family_signature']['readout']=='parallel flattened-grid and global-mean feature readouts'
    assert 'separate class-logit heads' in context['fingerprint']['output']
    altered={k:v.replace('def forward(self, images: torch.Tensor) -> torch.Tensor:', 'def forward(self, images: torch.Tensor) -> torch.Tensor:\n        images = images * images') for k,v in fixed.items()}
    assert altered != fixed and profile(altered,FASHION) is None


def test_inline_helpers_and_nonfinite_coefficients_cannot_reuse_reviewed_templates():
    feature_views=_additional_inference_source(53)
    assert profile(feature_views,FASHION)
    changed={k:v.replace('torch.flip(features, dims=(3,))', 'torch.flip(features, dims=(2,))') for k,v in feature_views.items()}
    assert changed!=feature_views and profile(changed,FASHION) is None
    original=_inference_source(39)
    infinite={k:v.replace('0.02 * torch.tanh','1e999 * torch.tanh') for k,v in original.items()}
    assert infinite!=original and profile(infinite,FASHION) is None


def test_coefficient_normalization_cannot_collapse_argmax_dependent_routes():
    source=_additional_inference_source(46)
    identical={k:v.replace('0.5380733489990235', '0.5380729675292969').replace('0.4619266510009766', '0.4619270324707031') for k,v in source.items()}
    assert identical!=source and profile(identical,FASHION) is None
    proportional={k:v.replace('0.5380733489990235', '1.0761459350585938').replace('0.4619266510009766', '0.9238540649414062') for k,v in source.items()}
    assert proportional!=source and profile(proportional,FASHION) is None


def test_core_bound_inference_templates_cannot_claim_unknown_view_parameters():
    refs=json.loads((Path(__file__).parents[1]/'experiments/ontology_fashion_inference_references.json').read_text(encoding='utf8'))['reviewed_wrappers']
    ref=next(r for r in refs if r.get('custom_wrapper_audit_index')==11)
    assert ref['composition_only']
    base=cnn(extra='self.flip_mix_logits = 0.5')['train.py'].replace('def forward(self, images):','def _forward_once(self, images):')
    base=base[:base.index('def build_model')]
    source={'train.py':base+'\n'+'\n\n'.join(textwrap.indent(s,'    ') for s in ref['source_methods'].values())}
    assert profile(source,FASHION) is None


def test_uniform_probability_map_is_fixed_inference_but_margin_routing_changes():
    refs=json.loads((Path(__file__).parents[1]/'experiments/ontology_fashion_inference_references.json').read_text(encoding='utf8'))['reviewed_wrappers']
    fixed=next(i for i,r in enumerate(refs) if r.get('custom_wrapper_audit_index')==32)
    margin=next(i for i,r in enumerate(refs) if r.get('custom_wrapper_audit_index')==3)
    result=review_pair(cnn(),_inference_source(fixed),FASHION)
    assert result['classification']=='preserving'
    assert 'p*(1+c*p)' in result['fingerprint']['aggregation']
    assert review_pair(cnn(),_inference_source(margin),FASHION)['classification']=='changing'


def test_pointwise_gate_mlp_activation_preserves_but_gate_or_basis_changes_do_not():
    from experiments.ontology_review_vision_lm import _fashion_gate_family
    gelu={'channel_gate':{'basis':['mean'],'operator':'affine GELU affine sigmoid'}}
    silu={'channel_gate':{'basis':['mean'],'operator':'affine SiLU affine sigmoid'}}
    assert _fashion_gate_family(gelu)==_fashion_gate_family(silu)
    ungated={'channel_gate':{'basis':['mean'],'operator':'affine SiLU affine'}}
    other_basis={'channel_gate':{'basis':['maximum'],'operator':'affine SiLU affine sigmoid'}}
    assert _fashion_gate_family(gelu)!=_fashion_gate_family(ungated)
    assert _fashion_gate_family(gelu)!=_fashion_gate_family(other_basis)
    tanh_gain={'channel_gate':{'basis':['mean'],'operator':'affine pointwise affine tanh gain'}}
    sigmoid_gate={'channel_gate':{'basis':['mean'],'operator':'affine pointwise affine sigmoid'}}
    assert _fashion_gate_family(tanh_gain)!=_fashion_gate_family(sigmoid_gate)


def test_fixed_window_arithmetic_must_not_access_signal_or_execute_helpers():
    good=ast.parse('''def windows(self,config):
    pattern=config.window_pattern.upper()
    sizes=[]
    for layer in range(config.n_layer):
        size=config.sequence_len if pattern[layer % len(pattern)] == 'L' else config.sequence_len // 4
        sizes.append((size,0))
    return sizes
''').body[0]
    assert _fixed_windows(good)
    bad=ast.parse('def windows(self,config):\n return self.router(config)').body[0]
    assert not _fixed_windows(bad)


def _catalog_lm(overrides=None):
    catalog=json.loads((Path(__file__).parents[1]/'experiments/ontology_vision_lm_references.json').read_text(encoding='utf8'))['nanogpt_method_catalog']
    selected={scope:next(iter(entries.values()))['source'] for scope,entries in catalog.items() if scope not in {'apply_partial_rotary_emb','has_context_gate'}}
    selected.update(overrides or {})
    selected['GPT._compute_window_sizes']='def _compute_window_sizes(self,config):\n return [(config.sequence_len//2,0) for i in range(config.n_layer)]'
    classes={}
    funcs=[]
    for scope,source in selected.items():
        if '.' in scope:classes.setdefault(scope.split('.')[0],[]).append(textwrap.indent(source,'    '))
        else:funcs.append(source)
    return {'train.py':'\n\n'.join(funcs+['class '+name+'(nn.Module):\n'+'\n'.join(methods) for name,methods in classes.items()])}


def test_reviewed_attention_gate_is_a_boundary_in_both_directions():
    catalog=json.loads((Path(__file__).parents[1]/'experiments/ontology_vision_lm_references.json').read_text(encoding='utf8'))['nanogpt_method_catalog']
    attention=next(e['source'] for e in catalog['CausalSelfAttention.forward'].values() if 'out_gate = 2 * torch.sigmoid(self.out_gate(' in e['source'] and 'if self.out_gate' not in e['source'])
    ctor=next(e['source'] for e in catalog['CausalSelfAttention.__init__'].values() if 'self.out_gate = nn.Linear(self.ve_gate_channels' in e['source'] and 'is_long_layer' not in e['source'])
    before=_catalog_lm();after=_catalog_lm({'CausalSelfAttention.forward':attention,'CausalSelfAttention.__init__':ctor})
    assert review_pair(before,after,NANO)['classification']=='changing'
    assert review_pair(after,before,NANO)['classification']=='changing'
    assert 'attention-output gate' in profile(after,NANO)['fingerprint']['routing']
    unsupported={k:v.replace('out_gate = 2 * torch.sigmoid','out_gate = 2 * torch.sin') for k,v in after.items()}
    assert profile(unsupported,NANO) is None


def test_grouped_query_weight_sharing_preserves_family_but_remains_in_fingerprint():
    catalog=json.loads((Path(__file__).parents[1]/'experiments/ontology_vision_lm_references.json').read_text(encoding='utf8'))['nanogpt_method_catalog']
    factory=next(e['source'] for e in catalog['build_model_config'].values() if 'n_kv_head=max(1, num_heads // 2)' in e['source'])
    before=_catalog_lm();after=_catalog_lm({'build_model_config':factory})
    constants='DEPTH=8\nASPECT_RATIO=64\nHEAD_DIM=128\nWINDOW_PATTERN="SSSL"\n'
    after={k:constants+v for k,v in after.items()}
    result=review_pair(before,after,NANO)
    assert result['classification']=='preserving'
    assert 'share each key/value head' in result['fingerprint']['projection_relations']


def test_convolution_connectivity_is_a_preserving_coefficient_restriction():
    dense=cnn()
    grouped=cnn(groups=2)
    result=review_pair(dense,grouped,FASHION)
    assert result['classification']=='preserving'
    assert result['fingerprint']['spatial_operator']!=profile(dense,FASHION)['fingerprint']['spatial_operator']
    assert result['settings']['convolution_connectivity']=='grouped'


def test_fixed_causal_head_masks_preserve_but_new_information_direction_is_unreviewed():
    catalog=json.loads((Path(__file__).parents[1]/'experiments/ontology_vision_lm_references.json').read_text(encoding='utf8'))['nanogpt_method_catalog']
    forward=next(e['source'] for e in catalog['CausalSelfAttention.forward'].values() if 'y_local = fa3.flash_attn_func' in e['source'])
    ctor=next(e['source'] for e in catalog['CausalSelfAttention.__init__'].values() if 'self.n_local_head' in e['source'])
    before=_catalog_lm();after=_catalog_lm({'CausalSelfAttention.forward':forward,'CausalSelfAttention.__init__':ctor})
    result=review_pair(before,after,NANO)
    assert result['classification']=='preserving'
    assert 'parallel local-window and global-window head groups' in result['fingerprint']['context_topology']
    future={k:v.replace('causal=True','causal=False') for k,v in after.items()}
    assert profile(future,NANO) is None


def test_value_gate_channel_width_preserves_the_existing_gating_primitive():
    catalog=json.loads((Path(__file__).parents[1]/'experiments/ontology_vision_lm_references.json').read_text(encoding='utf8'))['nanogpt_method_catalog']
    forward=next(e['source'] for e in catalog['CausalSelfAttention.forward'].values() if 'v = v + gate * ve' in e['source'])
    ctor=next(e['source'] for e in catalog['CausalSelfAttention.__init__'].values() if 'nn.Linear(self.ve_gate_channels, self.n_kv_head * self.head_dim' in e['source'])
    result=review_pair(_catalog_lm(),_catalog_lm({'CausalSelfAttention.forward':forward,'CausalSelfAttention.__init__':ctor}),NANO)
    assert result['classification']=='preserving'
    assert 'per-channel token-value gate' in result['fingerprint']['kv_memory']


def test_position_free_head_preserves_rotary_primitive_but_cannot_remove_all_rotary_heads():
    catalog=json.loads((Path(__file__).parents[1]/'experiments/ontology_vision_lm_references.json').read_text(encoding='utf8'))['nanogpt_method_catalog']
    forward=next(e['source'] for e in catalog['CausalSelfAttention.forward'].values() if 'self.use_nope_head' in e['source'])
    ctor=next(e['source'] for e in catalog['CausalSelfAttention.__init__'].values() if 'self.use_nope_head' in e['source'])
    helper=next(e['source'] for e in catalog['apply_rotary_emb'].values() if 'positionless_head=False' in e['source'])
    constants='DEPTH=8\nASPECT_RATIO=64\nHEAD_DIM=128\nWINDOW_PATTERN="SSSL"\n'
    before={k:constants+v for k,v in _catalog_lm().items()}
    after={k:constants+v for k,v in _catalog_lm({'CausalSelfAttention.forward':forward,'CausalSelfAttention.__init__':ctor,'apply_rotary_emb':helper}).items()}
    result=review_pair(before,after,NANO)
    assert result['classification']=='preserving'
    assert 'position-free' in result['fingerprint']['position']
    degenerate={k:v.replace('HEAD_DIM=128','HEAD_DIM=512') for k,v in after.items()}
    assert profile(degenerate,NANO) is None


def test_prefix_memory_keeps_its_new_basis_without_a_rank_only_family_key():
    catalog=json.loads((Path(__file__).parents[1]/'experiments/ontology_vision_lm_references.json').read_text(encoding='utf8'))['nanogpt_method_catalog']
    forward=next(e['source'] for e in catalog['GPT.forward'].values() if 'prefix_sum = x.cumsum' in e['source'])
    ctor=next(e['source'] for e in catalog['GPT.__init__'].values() if 'self.context_memory_down' in e['source'])
    result=review_pair(_catalog_lm(),_catalog_lm({'GPT.forward':forward,'GPT.__init__':ctor}),NANO)
    assert result['classification']=='changing'
    assert 'bottleneck' not in result['family_signature']
    assert 'low-rank projection' in result['fingerprint']['bottleneck']
    assert 'within-sequence prefix-sum/count representation' in result['family_signature']['state']


def test_swiglu_updates_all_relevant_fingerprint_components():
    catalog=json.loads((Path(__file__).parents[1]/'experiments/ontology_vision_lm_references.json').read_text(encoding='utf8'))['nanogpt_method_catalog']
    forward=next(e['source'] for e in catalog['MLP.forward'].values() if 'F.silu(gate)' in e['source'])
    ctor=next(e['source'] for e in catalog['MLP.__init__'].values() if 'nn.Linear(config.n_embd, 2 * hidden_dim' in e['source'])
    before=_catalog_lm();after=_catalog_lm({'MLP.forward':forward,'MLP.__init__':ctor})
    result=review_pair(before,after,NANO)
    assert result['classification']=='changing'
    assert 'SwiGLU' in result['fingerprint']['feedforward']
    assert 'squared ReLU MLP' not in result['fingerprint']['activation']
    assert 'SwiGLU' in result['fingerprint']['block_composition']
    bad=ast.parse('def windows(self,config):\n return secret_helper(config)').body[0]
    assert not _fixed_windows(bad)


def test_b05_c1_adaptive_pooling_pairs_have_exact_bindings_and_reject_mutations():
    catalog = json.loads((Path(__file__).parents[1] / 'experiments/ontology_fashion_candidate_references_v3_unresolved.json').read_text(encoding='utf8'))
    expected = {
        ('e145e2648f1a5132a61fd58a013033091d1e1d48ff538bfe818983d353f0f044', 'd0c5bb18ac9e1ab657b0e156df097cb159db51b947ced37c1f03b9f862cbfd15'),
        ('e145e2648f1a5132a61fd58a013033091d1e1d48ff538bfe818983d353f0f044', 'bcdcc6c3a5a9804e34d3a8c3066181fdb3fba0335b830dbc0500d42f49613b5a'),
    }
    rows = {(row['parent_source_sha256'], row['source_sha256']): row for row in catalog['reviewed_profiles']}
    runs = WORKSPACE / 'data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign/runs'
    assert expected <= rows.keys()
    for key in expected:
        row = rows[key]
        parent = source(runs / row['reference_run'] / 'candidates' / row['parent_candidate_id'])
        child = source(runs / row['reference_run'] / 'candidates' / row['candidate_id'])
        result = review_pair(parent, child, FASHION)
        assert result['classification'] == 'changing'
        assert result['changed_components'] == ['aggregation', 'scale_representation', 'spatial_readout']
        assert len(result['fingerprint']) == 27
        assert all(result['fingerprint'].values())
        evidence = row['evidence'][0]
        assert evidence['parent_source_sha256'] == key[0]
        assert evidence['source_sha256'] == key[1]
        assert 'AdaptiveAvgPool2d((4, 4))' in evidence['code']
        mutated = {'train.py': child['train.py'].replace('AdaptiveAvgPool2d((4, 4))', 'AdaptiveAvgPool2d((5, 5))')}
        assert review_pair(parent, mutated, FASHION) is None


def test_b03_c3_parallel_local_dilated_pairs_have_exact_bindings_and_reject_mutations():
    catalog = json.loads((Path(__file__).parents[1] / 'experiments/ontology_fashion_candidate_references_v3_unresolved.json').read_text(encoding='utf8'))
    expected = {
        ('3996c476411a6e6b243945658e8ec73bfaa0780a8cfed671ac9c9e3aada2d9a5', 'a5de14f617c096214487a4e5c07f65c916c0a9c89a754c451918e000bb0fcd15'),
        ('3996c476411a6e6b243945658e8ec73bfaa0780a8cfed671ac9c9e3aada2d9a5', 'bb8b184cf6eafd251944533069137c98db55d3ce20d8f8ab3c87829fcd17c15d'),
    }
    rows = {(row['parent_source_sha256'], row['source_sha256']): row for row in catalog['reviewed_profiles']}
    runs = WORKSPACE / 'data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign/runs'
    assert expected <= rows.keys()
    for key in expected:
        row = rows[key]
        parent = source(runs / row['reference_run'] / 'candidates' / row['parent_candidate_id'])
        child = source(runs / row['reference_run'] / 'candidates' / row['candidate_id'])
        result = review_pair(parent, child, FASHION)
        assert result['classification'] == 'preserving'
        assert result['changed_components'] == []
        assert len(result['fingerprint']) == 27
        assert all(result['fingerprint'].values())
        assert 'parallel local/dilated' in result['fingerprint']['mixing']
        assert 'dilated 3x3 stride-2 context branch' in result['fingerprint']['spatial_operator']
        evidence = row['evidence'][0]
        assert evidence['parent_source_sha256'] == key[0]
        assert evidence['source_sha256'] == key[1]
        assert 'dilation=2' in evidence['code']
        assert 'torch.cat' in evidence['code']
        mutated_source = child['train.py'].replace('dilation=2', 'dilation=3', 1)
        assert mutated_source != child['train.py']
        assert review_pair(parent, {'train.py': mutated_source}, FASHION) is None
