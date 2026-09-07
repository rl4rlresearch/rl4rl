"""Counterexamples and source-derived semantic transitions for task reviewer."""
import pytest

from experiments.ontology_review_recurrent import review_pair, profile

KWS='openevolve_v21_tiny_kws_rnn'
HAR='uci_har_pareto_v21'


KWS_B01_C2_CONFIDENCE_CASES=(
    ('P133','e3744cef','fd2a2b17','aa4fb4f4df4e6b5960c99e282d7c2289ed6f3a74740c4d44e48cd6ef14b5bcf1','2d0809509cdf1364452af9254b124e1930a122057279998b8b8972b5586b30b8'),
    ('P134','fc5c05f7','3f056f42','95ae6eaf82b6dcd63375d1a341fa31d1c38c2e7cd2ede4c7635f2b62231aeebf','3527ca48d3a6517de6bc6feed38e82c14e3691c3ca137939965e44da9c74e679'),
    ('P135','f102dbd4','ff5d82ea','227ce9d48fef6e7c5ccd92d440afdadb37c199bceadba8b185d8b4397d131081','4208ab87ca9460450021f7c0a7713e2d6ffaa2ca917028430543b1974739e8a7'),
    ('P138','ff5d82ea','ff78fc24','4208ab87ca9460450021f7c0a7713e2d6ffaa2ca917028430543b1974739e8a7','c3b1104a07658bd1cf48daafcb5866c83fcb9ba2a998ab7d841f947e72cff31b'),
    ('P139','3f056f42','445bd98c','3527ca48d3a6517de6bc6feed38e82c14e3691c3ca137939965e44da9c74e679','0336a376ae2d5dec4552c9422154b4dff268e69b518b592c2635f0f8a82d32c6'),
    ('P140','fd2a2b17','a93428b8','2d0809509cdf1364452af9254b124e1930a122057279998b8b8972b5586b30b8','7968eaf92900889f75a7f0d183c5c97a89de5f6b358d347f789cf5e4c2a2bfaf'),
    ('P141','a93428b8','236372e1','7968eaf92900889f75a7f0d183c5c97a89de5f6b358d347f789cf5e4c2a2bfaf','38c016519c0e8657d8dbdcc24e73102358b0fabb09dc2559849c49bd24cb6444'),
    ('P143','ff78fc24','8da9d2f0','c3b1104a07658bd1cf48daafcb5866c83fcb9ba2a998ab7d841f947e72cff31b','a46165e36c6b19d95b4215207716b1a737f5475e67550ed1a9a0f06b764ab08a'),
    ('P144','445bd98c','e9d8e26a','0336a376ae2d5dec4552c9422154b4dff268e69b518b592c2635f0f8a82d32c6','f67938866248e696de565d0264a77e2ce1c45a8399ae6974bd082fa643f09282'),
    ('P145','236372e1','46465b93','38c016519c0e8657d8dbdcc24e73102358b0fabb09dc2559849c49bd24cb6444','9c64a59865611c1eb9f8608de5e6b41e43affc9eba42d798025c93dab36bfbac'),
    ('P147','8da9d2f0','6fcac99d','a46165e36c6b19d95b4215207716b1a737f5475e67550ed1a9a0f06b764ab08a','c5076bd877a6653c1f37c1386d96bc1c6939c2e6c8a1d747e782750067656d34'),
    ('P148','e9d8e26a','ef243111','f67938866248e696de565d0264a77e2ce1c45a8399ae6974bd082fa643f09282','481a1670cbc9e17d49a2c23d653ccd071773aeda66e28dd7900c1d222296f93d'),
    ('P149','ef243111','217782c0','481a1670cbc9e17d49a2c23d653ccd071773aeda66e28dd7900c1d222296f93d','284687a1b657f3a773762e2dff8e4e0baaad9504e0eaca57c322d5270814f3eb'),
    ('P150','236372e1','46b55ddc','38c016519c0e8657d8dbdcc24e73102358b0fabb09dc2559849c49bd24cb6444','449b6ba74046c7879369f2b82b331a7a2be46d101041683fc401cdfbf416c670'),
    ('P152','8da9d2f0','6ea6e3f5','a46165e36c6b19d95b4215207716b1a737f5475e67550ed1a9a0f06b764ab08a','ab413512e0f7b99cee9fe0768eaa29e3c2ae06ac1fd46f10007b49ae5d14f6da'),
    ('P154','6ea6e3f5','ca29e4bf','ab413512e0f7b99cee9fe0768eaa29e3c2ae06ac1fd46f10007b49ae5d14f6da','723f2d0d537879c57748cdc4a54eec1556257ced6313a0ef87cc13b61f4456c3'),
    ('P155','217782c0','e45af576','284687a1b657f3a773762e2dff8e4e0baaad9504e0eaca57c322d5270814f3eb','c44af8cdd4d96301c15d72270585dd3d7966e4a41f62138c0ff329193df7a084'),
    ('P156','46b55ddc','2dcaf7b1','449b6ba74046c7879369f2b82b331a7a2be46d101041683fc401cdfbf416c670','65116b765a0d12c8d0f2ae9dc6e3ff3c93cc92280343bbd1fe5b641c8e4696d7'),
    ('P157','e45af576','cc9790a7','c44af8cdd4d96301c15d72270585dd3d7966e4a41f62138c0ff329193df7a084','78e46556470466eef54f9ae709078c2a0e18aa5be8b51a217db3180ad909dac4'),
    ('P160','46b55ddc','adb0aa46','449b6ba74046c7879369f2b82b331a7a2be46d101041683fc401cdfbf416c670','be0c4a7b368089c2a79124f040fa2d5a883393a6b320609dca4eb97039873fea'),
    ('P161','e45af576','bfb3a04e','c44af8cdd4d96301c15d72270585dd3d7966e4a41f62138c0ff329193df7a084','7ca19048bbb39b3d194cf6850fbd44a00f2a8b2f51d80a5031527744347b8177'),
    ('P163','ca29e4bf','b4358704','723f2d0d537879c57748cdc4a54eec1556257ced6313a0ef87cc13b61f4456c3','6ac267c7dac30725f6d82cd0a583e2f4c36d358e71cf53de4ca73bc8e6325147'),
    ('P164','46b55ddc','912075a7','449b6ba74046c7879369f2b82b331a7a2be46d101041683fc401cdfbf416c670','6e75e57d7537e0bb47cadf04e254bfbee6efa2acff13ba6163ce08e7906fab01'),
    ('P165','e45af576','56f43173','c44af8cdd4d96301c15d72270585dd3d7966e4a41f62138c0ff329193df7a084','0f561412026ca710fc887e44d1a4d3efee1283532fa95cbb4dee3143adbf4bc6'),
    ('P167','ca29e4bf','253ad965','723f2d0d537879c57748cdc4a54eec1556257ced6313a0ef87cc13b61f4456c3','17a4acec7cda13b10b9a125dc0457410a85b7cc1647a05099ee6f478acbcd4c3'),
    ('P168','912075a7','7cdc6229','6e75e57d7537e0bb47cadf04e254bfbee6efa2acff13ba6163ce08e7906fab01','7b4c10d699d9bb2b81399c8250c754ba9cd1c26dbf1ae110c83ec345f139c180'),
    ('P169','e45af576','f2b2781f','c44af8cdd4d96301c15d72270585dd3d7966e4a41f62138c0ff329193df7a084','82ceddef8fb674f28fb13eec7c1503a328a1abbaf180cddc32582e75ef58d78a'),
)


def _fixture_repository_root():
    """Locate campaign fixtures when pytest runs from an immutable engine."""
    from pathlib import Path
    return next(
        parent for parent in Path(__file__).resolve().parents
        if (parent/'data'/'c0c3').is_dir()
    )

BASE='''
import torch
from torch import nn
class KeywordGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(20)
        self.gru=nn.GRU(20,80,batch_first=True)
        self.head=nn.Linear(80,8)
    def initial_state(self,batch_size,device,dtype):
        return torch.zeros(batch_size,1,80),torch.zeros(batch_size,80),torch.zeros(batch_size,1)
    def recurrent_step(self,frame,state):
        hidden,total,count=state
        output,hidden=self.gru(self.norm(frame).unsqueeze(1),hidden.transpose(0,1))
        output=output[:,0,:]
        return hidden.transpose(0,1),total+output,count+1.0
    def classify(self,state):
        hidden,total,count=state
        return self.head(total/count.clamp_min(1.0))
    def frame_schedule(self,available_frames):
        return list(range(available_frames))
def build_model():return KeywordGRU()
'''


def review(a,b,key=KWS):return review_pair({'train.py':a},{'train.py':b},key)


def test_width_and_numeric_schedule_are_preserving():
    answer=review(BASE,BASE.replace('80','64').replace('range(available_frames)','range(0,available_frames,2)'))
    assert answer['classification']=='preserving'
    assert answer['fingerprint_complete']


def test_har_microbi_ranked_pruning_slots_have_an_exact_preserving_proof():
    root = _fixture_repository_root() / 'data' / 'c0c3' / 'uci-har-pareto-v21' / 'runs'
    run = root / 'uci-har-pareto-v21-uci-har-pareto-openevolve-b02-c0' / 'candidates'
    parent = 'c62d39e616c8da6c988fbf89dfc83d778b55dde1b98b95a23b2668aacc8cd146'
    child = 'ef573f1e32c3ab91319c8bf27c98afa03cf78831e6aa31e2660c2279143bbd3e'
    answer = review_pair(
        {'train.py': (run / parent / 'train.py').read_text(encoding='utf-8')},
        {'train.py': (run / child / 'train.py').read_text(encoding='utf-8')},
        HAR,
    )
    assert answer['classification'] == 'preserving'
    assert answer['fingerprint_complete'] is True
    assert not answer['residual_components']
    assert answer['reviewer'] == 'har-microbi-ranked-pruning-exact-source-v1'


def test_har_microbi_pruning_proof_rejects_an_unreviewed_source_change():
    root = _fixture_repository_root() / 'data' / 'c0c3' / 'uci-har-pareto-v21' / 'runs'
    run = root / 'uci-har-pareto-v21-uci-har-pareto-openevolve-b02-c0' / 'candidates'
    source = (run / 'c62d39e616c8da6c988fbf89dfc83d778b55dde1b98b95a23b2668aacc8cd146' / 'train.py').read_text(encoding='utf-8')
    changed = source.replace('self.inferenceFeatures = self.summaryFeatures - 22', 'self.inferenceFeatures = self.summaryFeatures - 21')
    from experiments.ontology_har_microbi_pruning import review_pair as pruning_review
    assert pruning_review({'train.py': source}, {'train.py': changed}) is None


def test_add_final_recurrent_state_to_mean_is_new_history():
    after=BASE.replace('self.head(total/count.clamp_min(1.0))','self.head(total/count.clamp_min(1.0)+hidden[:,0,:])')
    answer=review(BASE,after)
    assert answer['classification']=='changing'
    assert 'readout_history' in answer['changed_components']
    assert answer['transition_evidence'] and answer['evidence']


def test_affine_gru_input_projection_is_not_a_new_state_equation():
    after=BASE.replace('self.gru=','self.project=nn.Linear(20,16)\n        self.gru=').replace('nn.GRU(20,80','nn.GRU(16,80').replace('self.norm(frame).unsqueeze(1)','self.project(self.norm(frame)).unsqueeze(1)')
    answer=review(BASE,after)
    assert answer['classification']=='preserving'


def test_nonlinear_projection_cannot_use_affine_absorption_proof():
    after=BASE.replace('self.gru=','self.project=nn.Linear(20,16)\n        self.gru=').replace('nn.GRU(20,80','nn.GRU(16,80').replace('self.norm(frame).unsqueeze(1)','torch.tanh(self.project(self.norm(frame))).unsqueeze(1)')
    answer=review(BASE,after)
    assert answer['classification']=='changing'
    assert 'acoustic_representation' in answer['changed_components']


def test_unused_constructor_does_not_create_a_recurrent_mechanism():
    after=BASE.replace('self.norm=','self.unused=nn.LSTM(20,80)\n        self.norm=')
    answer=review(BASE,after)
    assert answer is None or answer['classification']!='changing'


def test_variable_renaming_does_not_create_family_features():
    after=BASE.replace('hidden','memory').replace('total','running_sum').replace('count','n_seen')
    a=profile({'train.py':BASE},KWS);b=profile({'train.py':after},KWS)
    assert a['mechanism']==b['mechanism']


def test_unhandled_custom_computation_is_not_labelled_by_source_difference():
    before=BASE.replace('return self.head(total/count.clamp_min(1.0))','return external_engine(state)')
    after=before.replace('external_engine','different_external_engine')
    assert review(before,after) is None


def test_positive_recurrence_witness_does_not_claim_complete_custom_fingerprint():
    before=BASE.replace('return self.head(total/count.clamp_min(1.0))','return external_engine(state)')
    after=before.replace('nn.GRU(','nn.LSTM(')
    answer=review(before,after)
    assert answer['classification']=='changing'
    assert not answer['fingerprint_complete']
    assert 'classify' in answer['residual_components']


def test_source_code_is_never_executed(tmp_path):
    target=tmp_path/'candidate-ran'
    source='open('+repr(str(target))+',"w").write("bad")\n'+BASE
    review(source,source.replace('80','79'))
    assert not target.exists()


def test_threshold_regions_tune_existing_exit_family():
    method='''    def exit_mask(self,logits,state,step,total_steps):
        del step
        probabilities=logits.softmax(dim=-1)
        confidence=probabilities.max(dim=-1).values
        top=probabilities.topk(2,dim=-1).values
        margin=top[:,0]-top[:,1]
        confident=confidence>=0.5
        decisive=(confidence>=0.45)&(margin>=0.25)
        return confident|decisive
'''
    a=BASE.replace('def build_model()',method+'def build_model()')
    b=a.replace('return confident|decisive','extra=(confidence>=0.43)&(confidence<0.45)&(margin>=0.3)\n        return confident|decisive|extra')
    answer=review(a,b)
    assert answer['classification']=='preserving'


def test_exit_family_rejects_new_signal_observable():
    method='''    def exit_mask(self,logits,state,step,total_steps):
        return logits.softmax(dim=-1).max(dim=-1).values>0.5
'''
    a=BASE.replace('def build_model()',method+'def build_model()')
    b=a.replace('return logits.softmax(dim=-1).max(dim=-1).values>0.5','return (logits.softmax(dim=-1).max(dim=-1).values>0.5)&(state[0].abs().mean(dim=-1)>0.1)')
    answer=review(a,b)
    assert answer['classification']=='changing'
    assert 'exit_policy' in answer['changed_components']


def test_parameter_derived_pruning_does_not_create_new_history():
    from experiments.ontology_recurrent_readout import same_coordinate_readout
    a=BASE.replace('self.head=nn.Linear(80,8)','self.head=nn.Linear(80,8)\n        self.inference_head=nn.Linear(79,7)\n        self.register_buffer("indices",torch.arange(79))')
    a=a.replace('return self.head(total/count.clamp_min(1.0))','readout=total/count.clamp_min(1.0)\n        if self.training:\n            return self.head(readout)\n        contrasts=self.inference_head(readout.index_select(-1,self.indices))\n        return torch.cat((contrasts,contrasts.new_zeros(contrasts.shape[0],1)),dim=-1)')
    b=a.replace('79','78')
    assert same_coordinate_readout({'train.py':a},{'train.py':b})
    changed=b.replace('readout=total/count.clamp_min(1.0)','readout=total/count.clamp_min(1.0)+hidden[:,0,:]')
    assert not same_coordinate_readout({'train.py':a},{'train.py':changed})


def test_kws_deployment_coordinate_pruning_is_source_bound_and_complete():
    """The B01-C0 landmark handles a guarded parameter-ranked head only."""
    from pathlib import Path
    from experiments.ontology_review_recurrent import source_sha

    root = _fixture_repository_root() / (
        'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/'
        'tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b01-c0/candidates'
    )
    before = next(root.glob('eafba2393efa*'))
    after = next(root.glob('2382c7b9e5aa*'))
    read = lambda path: {
        file.relative_to(path).as_posix(): file.read_text(encoding='utf-8-sig')
        for file in path.rglob('*.py')
    }
    before_source, after_source = read(before), read(after)
    assert source_sha(before_source) == 'c8b002b806387653e94ef82395f862f54ac7cd5d01fb1185ca22c9923ffd0d23'
    assert source_sha(after_source) == 'e102e8f38de7488298b05c918e3897f3aca8efcb03f0fa10d6e9d418c80ef977'

    result = review_pair(before_source, after_source, KWS)
    assert result['classification'] == 'preserving'
    assert result['fingerprint_complete']
    assert not result['residual_components']
    assert len(result['fingerprint']) == 27
    assert result['reviewer'] == 'recurrent-kws-b01-c0-coordinate-pruning-direct-source-v1'
    assert 'parameter-ranked existing readout coordinates' in result['notes']

    other_parent = dict(before_source)
    other_parent['train.py'] = other_parent['train.py'].replace('BATCH_SIZE = 64', 'BATCH_SIZE = 65')
    mismatch = review_pair(other_parent, after_source, KWS)
    assert mismatch['before_source_sha256'] == source_sha(other_parent)
    assert mismatch['reviewer'] != 'recurrent-kws-b01-c0-coordinate-pruning-direct-source-v1'


def test_kws_coordinate_pruning_parser_covers_later_same_shape_sources():
    """Numeric pruning-width changes retain a complete source fingerprint."""
    from pathlib import Path
    from experiments.ontology_review_recurrent import _fingerprint

    root = _fixture_repository_root() / (
        'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/'
        'tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b01-c0/candidates'
    )
    for candidate in ('10b7ebffc2fc', '4f98b8ba3ea9'):
        path = next(root.glob(candidate + '*'))
        source = {
            file.relative_to(path).as_posix(): file.read_text(encoding='utf-8-sig')
            for file in path.rglob('*.py')
        }
        parsed = profile(source, KWS)
        fingerprint, residual = _fingerprint(parsed)
        assert parsed['coordinate_pruning'] is not None
        assert parsed['errors'] == {}
        assert len(fingerprint) == 27
        assert residual == []


def test_coordinate_pruning_contract_rejects_stochastic_selection():
    from experiments.ontology_recurrent_readout import coordinate_pruning_contract

    source = BASE.replace(
        'self.head=nn.Linear(80,8)',
        'self.head=nn.Linear(80,8)\n        self.inference_head=nn.Linear(79,7)\n        self.register_buffer("indices",torch.arange(79))',
    )
    source = source.replace(
        '    def initial_state',
        '    def train(self,mode=True):\n        super().train(mode)\n        if not mode:\n            selected=torch.rand(80).topk(79).indices\n            self.indices.copy_(selected)\n        return self\n    def initial_state',
    ).replace(
        'return self.head(total/count.clamp_min(1.0))',
        'readout=total/count.clamp_min(1.0)\n        if self.training:\n            return self.head(readout)\n        contrasts=self.inference_head(readout.index_select(-1,self.indices))\n        return torch.cat((contrasts,contrasts.new_zeros(contrasts.shape[0],1)),dim=-1)',
    )
    assert coordinate_pruning_contract({'train.py': source}) is None


def test_branch_extension_does_not_call_count_delay_a_new_mechanism():
    from experiments.ontology_recurrent_branches import review_pair as branch_review
    a=BASE.replace('output,hidden=self.gru','if bool(torch.all(count==0)):\n            return hidden,total,count+1.0\n        output,hidden=self.gru')
    b=a.replace('count==0','count<2')
    answer=branch_review({'train.py':a},{'train.py':b},KWS)
    assert answer is None or answer['classification']!='changing'


def test_branch_extension_retains_signal_gating():
    from experiments.ontology_recurrent_branches import review_pair as branch_review
    a=BASE.replace('output,hidden=self.gru','if bool(torch.all(count==0)):\n            return hidden,total,count+1.0\n        output,hidden=self.gru')
    b=a.replace('torch.all(count==0)','torch.all(frame.abs().mean(dim=-1)>0.1)')
    answer=branch_review({'train.py':a},{'train.py':b},KWS)
    assert answer['classification']=='changing'
    assert 'state_structure' in answer['changed_components']


def test_accounted_equivalence_resolves_variable_renames():
    b=BASE.replace('hidden','memory').replace('total','running_sum').replace('count','n_seen')
    answer=review(BASE,b)
    assert answer['classification']=='preserving'
    assert answer['fingerprint_complete']


def test_accounted_equivalence_does_not_erase_layout_axis_changes():
    from experiments.ontology_recurrent_equivalence import review_pair as equation_review
    b=BASE.replace('hidden.transpose(0,1)','hidden.transpose(0,2)')
    assert equation_review({'train.py':BASE},{'train.py':b},KWS) is None


def test_main_dispatch_invokes_branch_extension():
    a=BASE.replace('output,hidden=self.gru','if bool(torch.all(count==0)):\n            return hidden,total,count+1.0\n        output,hidden=self.gru')
    b=a.replace('torch.all(count==0)','torch.all(frame.abs().mean(dim=-1)>0.1)')
    answer=review(a,b)
    assert answer['classification']=='changing'
    assert answer['reviewer']=='recurrent-branch-v1'


def test_library_to_manual_recurrence_is_not_assumed_changing():
    # An external/fused/manual cell needs its equations reviewed. Mere absence
    # of a standard GRU declaration cannot distinguish that from a new cell.
    after=BASE.replace('self.gru=nn.GRU(20,80,batch_first=True)','self.gru=ManualGRU(20,80)')
    answer=review(BASE,after)
    assert answer is None or answer['classification']!='changing'


def test_shared_tensor_graph_is_serialized_without_exponential_expansion():
    from experiments.ontology_review_recurrent import _json,_walk,_pretty
    node=('signal',0)
    for _ in range(40):node=('tanh',('Add',node,node))
    assert len(_json(node))<12000
    assert len(list(_walk(node)))==81
    assert len(_pretty(node))<12000


def test_fixed_initial_state_comprehension_and_starred_tuple_are_accounted():
    b=BASE.replace('return torch.zeros(batch_size,1,80),torch.zeros(batch_size,80),torch.zeros(batch_size,1)','hidden=torch.zeros(batch_size,1,80)\n        accumulators=tuple(torch.zeros(batch_size,80) for _ in range(1))\n        return hidden,*accumulators,torch.zeros(batch_size,1)')
    p=profile({'train.py':b},KWS)
    assert 'initial_state' not in p['errors']
    assert p['equations']['initial_state'][0]=='tuple'


def test_comprehension_never_invokes_an_unknown_iterator():
    b=BASE.replace('return torch.zeros(batch_size,1,80),torch.zeros(batch_size,80),torch.zeros(batch_size,1)','return tuple(torch.zeros(batch_size,80) for _ in external_iterator())')
    p=profile({'train.py':b},KWS)
    assert 'initial_state' in p['errors']


def test_repeating_an_existing_pool_does_not_create_new_ontology():
    from experiments.ontology_review_recurrent import _feature_witnesses
    features=('convolution-features','Conv1d','dense')
    a=('inputs',('pool','MaxPool1d',features))
    b=('inputs',('pool','MaxPool1d',('pool','MaxPool1d',features)))
    assert _feature_witnesses(a,b,'input_representation')==[]
    c=('inputs',('pool','AvgPool1d',features))
    assert _feature_witnesses(a,c,'input_representation')


def test_readout_coordinate_pruning_is_preserving_not_new_history():
    b=BASE.replace('self.head=nn.Linear(80,8)','self.head=nn.Linear(79,8)').replace('self.head(total/count.clamp_min(1.0))','self.head(total[:,:-1]/count.clamp_min(1.0))')
    answer=review(BASE,b)
    assert answer['classification']=='preserving'
    assert not answer['changed_components']


def test_direct_source_review_requires_both_exact_hashes(monkeypatch):
    import experiments.ontology_review_recurrent as module
    a={'train.py':BASE};b={'train.py':BASE.replace('80','79')}
    ref={'campaign':KWS,'parent_source_sha256':module.source_sha(a),'source_sha256':module.source_sha(b),'classification':'preserving','notes':'Hash-bound reviewed pair','evidence':[{'file':'train.py','line':1,'code':'source diff'}]}
    monkeypatch.setattr(module,'_direct_references',lambda:[ref])
    answer=module.review_pair(a,b,KWS)
    assert answer['reviewer']=='recurrent-direct-source-review-v2'
    changed={'train.py':b['train.py'].replace('self.head(total/count.clamp_min(1.0))','self.head(total/count.clamp_min(1.0)+hidden[:,0,:])')}
    answer=module.review_pair(a,changed,KWS)
    assert answer['classification']=='changing'
    assert answer['reviewer']!='recurrent-direct-source-review-v2'


def test_manual_full_fingerprint_does_not_transfer_transition_to_another_parent(monkeypatch):
    import experiments.ontology_review_recurrent as module
    a={'train.py':BASE}
    b={'train.py':BASE.replace('self.head(total/count.clamp_min(1.0))','self.head(total/count.clamp_min(1.0)+hidden[:,0,:])')}
    fp,residual=module._fingerprint(module.profile(b,KWS))
    assert not residual
    ref={'campaign':KWS,'parent_source_sha256':module.source_sha(a),'source_sha256':module.source_sha(b),
         'classification':'changing','changed_components':['readout_history'],'notes':'Reviewed extra final-state readout',
         'fingerprint':fp,'fingerprint_complete':True,'family_signature':{'history':'mean+final'},
         'implemented':True,'executable':False,'execution_diagnostic':{'kind':'test_shape_mismatch'},
         'evidence':[{'file':'train.py','line':1,'code':b['train.py']}]}
    monkeypatch.setattr(module,'_direct_references',lambda:[ref])
    direct=module.review_pair(a,b,KWS)
    assert direct['classification']=='changing'
    assert direct['changed_components']==['readout_history']
    assert direct['executable'] is False
    assert direct['execution_diagnostic']=={'kind':'test_shape_mismatch'}
    same=module.review_pair(b,b,KWS)
    assert same['classification']=='preserving'
    assert same['fingerprint_complete']
    assert same['family_signature']=={'history':'mean+final'}
    assert same['executable'] is False


def test_manual_complete_flag_requires_every_component():
    import experiments.ontology_review_recurrent as module
    ref={'fingerprint_complete':True,'fingerprint':{'state_update':'GRU'},'evidence':[{'code':'source'}]}
    assert module._reviewed_fingerprint(ref,KWS) is None


def test_tensor_method_arithmetic_matches_operator_equations():
    from experiments.ontology_review_recurrent import Equation
    import ast
    equation=Equation({}, {}, {})
    equation.values={'x':('signal',0),'y':('signal',1)}
    for method,operator in [('add','+'),('sub','-'),('mul','*'),('div','/')]:
        assert equation.expr(ast.parse(f'x.{method}(y)',mode='eval').body)==equation.expr(ast.parse(f'x{operator}y',mode='eval').body)
    assert equation.expr(ast.parse('x.add(y,alpha=0)',mode='eval').body)!=equation.expr(ast.parse('x+y',mode='eval').body)


def test_tensor_comparison_retains_observables_for_exit_review():
    from experiments.ontology_review_recurrent import Equation
    import ast
    equation=Equation({}, {}, {});equation.values={'x':('signal',0),'y':('signal',1)}
    assert equation.expr(ast.parse('x.ge(y)',mode='eval').body)==equation.expr(ast.parse('x>=y',mode='eval').body)
    assert equation.expr(ast.parse('x.ge(y)',mode='eval').body)!=equation.expr(ast.parse('x.ge(x)',mode='eval').body)


def test_shape_star_expansion_is_bounded_to_known_tensor_allocations():
    from experiments.ontology_review_recurrent import Equation,Unsupported
    import ast,pytest
    equation=Equation({}, {}, {});equation.values={'x':('signal',0)}
    assert equation.expr(ast.parse('torch.zeros(*x.shape)',mode='eval').body)==('initial-state','zeros')
    with pytest.raises(Unsupported):equation.expr(ast.parse('unknown(*x.shape)',mode='eval').body)


def test_affine_head_review_preserves_unmodified_custom_cell_sources():
    from experiments.ontology_recurrent_readout import same_coordinate_readout
    cell='''\nclass ExplicitGRU(nn.Module):
    def __init__(self,input_size,hidden_size,batch_first=True):
        super().__init__()
        self.cell=nn.GRU(input_size,hidden_size,batch_first=batch_first)
    def forward(self,x,h):
        return self.cell(x,h)
'''
    a=BASE.replace('self.gru=nn.GRU','self.gru=ExplicitGRU')+cell
    b=a.replace('self.head=nn.Linear(80,8)','self.head=nn.Linear(79,8)').replace('self.head(total/count.clamp_min(1.0))','self.head(total[:,:-1]/count.clamp_min(1.0))')
    assert same_coordinate_readout({'train.py':a},{'train.py':b})
    changed=b.replace('return self.cell(x,h)','return self.cell(x.square(),h)')
    assert not same_coordinate_readout({'train.py':a},{'train.py':changed})


def test_plain_recurrent_sequence_fold_is_accounted_without_running_frames():
    from experiments.ontology_recurrent_branches import branch_profile,is_plain_sequence_fold
    import ast
    method='''def recurrent_sequence(self,frames,state):
    for index in range(frames.shape[1]):
        state=self.recurrent_step(frames[:,index,:],state)
    return state
'''
    assert is_plain_sequence_fold(ast.parse(method).body[0])
    assert not is_plain_sequence_fold(ast.parse(method.replace('frames[:,index,:]','frames[:,index,:].square()')).body[0])
    assert not is_plain_sequence_fold(ast.parse(method.replace('range(frames.shape[1])','range(0,frames.shape[1],2)')).body[0])
    b=BASE.replace('    def classify', '\n'.join('    '+line for line in method.splitlines())+'\n    def classify')
    p=branch_profile({'train.py':b},KWS)
    assert 'recurrent_sequence' not in p['errors']
    assert p['equations']['recurrent_sequence'][0]=='causal-fold'


def test_fixed_output_coordinate_list_assembly_is_source_accounted():
    import ast
    from experiments.ontology_review_recurrent import Equation,_walk
    method=ast.parse('''def classify(self,x):
    scaled=[x[:,index] for index in range(3)]
    logits=[sum(scaled)]
    logits.extend(sum(scaled[row:])-row*scaled[row-1] for row in range(1,3))
    logits.append(-3.0*scaled[2])
    return torch.stack(logits,dim=-1)
''').body[0]
    eq=Equation({}, {}, {}).run(method)
    assert eq[0]=='combine' and len(eq)==5
    assert any(n[0]=='Sub' for n in _walk(eq) if isinstance(n,tuple))


def test_local_collection_assembly_rejects_aliases_and_dynamic_extensions():
    import ast,pytest
    from experiments.ontology_review_recurrent import Equation,Unsupported
    template='''def forward(self,x):
    outputs=[x]
    INSERT
    outputs.append(x.square())
    return torch.stack(outputs,dim=0)
'''
    for statement in ['alias=outputs','alias=(outputs,)','other=self.helper(outputs)','outputs.extend(external_iterator())']:
        with pytest.raises(Unsupported):Equation({}, {}, {}).run(ast.parse(template.replace('INSERT',statement)).body[0])
    with pytest.raises(Unsupported):
        Equation({}, {}, {}).run(ast.parse('def forward(self,x):\n    x.append(x)\n    return x').body[0])


def test_literal_lag_loop_tensor_list_assembly_is_accounted():
    import ast
    from experiments.ontology_review_recurrent import Equation
    method=ast.parse('''def forward(self,x):
    correlations=[]
    for lag in (0,2,8):
        correlations.append(x[...,lag:].mean(dim=-1))
    return torch.cat(correlations,dim=1)
''').body[0]
    result=Equation({}, {}, {}).run(method)
    assert result[0]=='combine' and len(result)==4


def test_literal_lag_loop_rejects_dynamic_or_mutable_assembly():
    import ast,pytest
    from experiments.ontology_review_recurrent import Equation,Unsupported
    template='''def forward(self,x,lags):
    correlations=[]
    for lag in ITERABLE:
        BODY
    return torch.cat(correlations,dim=DIM)
'''
    rejected=[
        ('lags','correlations.append(x[...,lag:])','1'),
        ('(0,2)','if lag: correlations.append(x)\n        else: correlations.append(x.square())','1'),
        ('(0,2)','alias=correlations\n        correlations.append(x)','1'),
        ('(0,2)','correlations.append(x)','axis'),
    ]
    for iterable,body,axis in rejected:
        with pytest.raises(Unsupported):
            Equation({}, {}, {}).run(ast.parse(template.replace('ITERABLE',iterable).replace('BODY',body).replace('DIM',axis)).body[0])


def test_literal_lag_loop_accepts_only_ordered_temporary_then_append():
    import ast
    from experiments.ontology_review_recurrent import Equation
    method=ast.parse('''def forward(self,x):
    magnitudeCorrelations=[]
    for lag in (8,16,32):
        correlation=(x[:,lag:]*x[:,:-lag]).mean(dim=1)
        magnitudeCorrelations.append(correlation/(x.square().mean(dim=1)+1e-4))
    return torch.cat(magnitudeCorrelations,dim=1)
''').body[0]
    assert Equation({}, {}, {}).run(method)[0]=='combine'


def test_literal_lag_loop_rejects_temporary_reordering_or_escape():
    import ast,pytest
    from experiments.ontology_review_recurrent import Equation,Unsupported
    cases=['''def forward(self,x):
    values=[]
    for lag in (1,2):
        values.append(x)
        item=x.square()
    return torch.cat(values,dim=1)
''','''def forward(self,x):
    values=[]
    for lag in (1,2):
        item=x.square()
        values.append(item+item)
    return torch.cat(values,dim=1)
''']
    for source in cases:
        with pytest.raises(Unsupported):
            Equation({}, {}, {}).run(ast.parse(source).body[0])


def test_runtime_and_comprehension_tensor_selectors_keep_actual_dependencies():
    import ast
    from experiments.ontology_review_recurrent import Equation,_walk
    equation=Equation({}, {}, {});equation.values={'x':('signal',0),'choice':('argmax',('signal',1))}
    fixed=equation.expr(ast.parse('[x[:,index] for index in range(3)]',mode='eval').body)
    assert len(set(fixed[1:]))==3
    selected=equation.expr(ast.parse('x[:,choice]',mode='eval').body)
    assert ('argmax',('signal',1)) in list(_walk(selected))
    assert selected!=equation.expr(ast.parse('x[:,0]',mode='eval').body)


def test_feedforward_har_fingerprint_handles_empty_recurrent_operator_tuple():
    from experiments.ontology_review_recurrent import _pretty
    source='''import torch
from torch import nn
class SensorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv1d(9,32,3)
        self.head=nn.Linear(32,6)
    def forward(self,x):
        x=self.conv(x.transpose(1,2))
        return self.head(x.mean(dim=-1))
def build_model():return SensorModel()
'''
    answer=review(source,source,HAR)
    assert answer['classification']=='preserving' and answer['fingerprint_complete']
    assert 'no recurrent state' in answer['fingerprint']['state_update']
    node=()
    for _ in range(20):node=('tanh',node)
    assert 'empty tuple' in _pretty(node)


def test_fixed_helmert_contrast_reconstruction_is_preserving():
    after=BASE.replace('import torch','import math\nimport torch').replace('self.head=nn.Linear(80,8)','self.head=nn.Linear(80,7)').replace('return self.head(total/count.clamp_min(1.0))','''contrasts=self.head(total/count.clamp_min(1.0))
        scaled=[contrasts[:,index]/math.sqrt((index+1)*(index+2)) for index in range(7)]
        logits=[sum(scaled)]
        logits.extend(sum(scaled[row:])-row*scaled[row-1] for row in range(1,7))
        logits.append(-7.0*scaled[6])
        return torch.stack(logits,dim=-1)''')
    answer=review(BASE,after)
    assert answer['classification']=='preserving' and answer['fingerprint_complete']
    from experiments.ontology_recurrent_readout import same_coordinate_readout
    assert not same_coordinate_readout({'train.py':BASE},{'train.py':after.replace('return torch.stack(logits,dim=-1)','return torch.stack(logits,dim=-1).square()')})


def test_affine_output_proof_rejects_signal_selected_columns_and_signal_scaling():
    from experiments.ontology_recurrent_readout import affine_output_features
    head=('affine',('signal',0))
    fixed=('axes-index',('slice-index',None,None,None),('fixed-index','2'))
    dynamic=('axes-index',('slice-index',None,None,None),('computed-index',('argmax',('signal',1))))
    assert affine_output_features(('select',head,fixed))==('signal',0)
    assert affine_output_features(('select',head,dynamic)) is None
    batch_select=('axes-index',('fixed-index','2'),('slice-index',None,None,None))
    assert affine_output_features(('select',head,batch_select)) is None
    assert affine_output_features(('Div',head,('norm',head))) is None


def test_tensor_shape_prefix_star_is_only_a_shape_argument():
    import ast,pytest
    from experiments.ontology_review_recurrent import Equation,Unsupported
    equation=Equation({}, {}, {});equation.values={'x':('signal',0)}
    assert equation.expr(ast.parse('x.reshape(*x.shape[:-1],3,2)',mode='eval').body)==('signal',0)
    with pytest.raises(Unsupported):equation.expr(ast.parse('unknown(*x.shape[:-1])',mode='eval').body)


def test_functional_layer_normalization_is_recorded_in_full_fingerprint():
    after=BASE.replace('import torch','from torch.nn import functional as F\nimport torch').replace('self.norm(frame)','F.layer_norm(frame,(20,))')
    answer=review(after,after)
    assert answer['fingerprint_complete']
    assert 'F.layer_norm' in answer['fingerprint']['normalization']


def test_existing_frequency_pair_pool_extent_is_a_preserving_setting():
    helper='''    def _input_features(self,frame):
        normalized=self.norm(frame)
        pooled=normalized[...,16:].reshape(*normalized.shape[:-1],2,2).mean(dim=-1)
        return torch.cat((normalized[...,:16],pooled),dim=-1)
'''
    before=BASE.replace('self.norm(frame)','self._input_features(frame)').replace('    def initial_state',helper+'    def initial_state').replace('nn.GRU(20,80','nn.GRU(18,80')
    after=before.replace('16:', '14:').replace(':16]', ':14]').replace('shape[:-1],2,2','shape[:-1],3,2').replace('nn.GRU(18,80','nn.GRU(17,80')
    answer=review(before,after)
    assert answer['classification']=='preserving' and answer['fingerprint_complete']
    from experiments.ontology_recurrent_readout import same_frequency_pool_extent
    assert not same_frequency_pool_extent({'train.py':BASE},{'train.py':after})
    assert not same_frequency_pool_extent({'train.py':before},{'train.py':after.replace('mean(dim=-1)','mean(dim=-2)')})
    assert not same_frequency_pool_extent({'train.py':before},{'train.py':after.replace('shape[:-1],3,2','shape[:-1],2,3')})
    assert not same_frequency_pool_extent({'train.py':before},{'train.py':after.replace('self.head(total/count.clamp_min(1.0))','self.head(total/count.clamp_min(1.0)+hidden[:,0,:])')})


def test_whole_source_template_changes_only_declared_numeric_slots():
    from experiments.ontology_review_recurrent import reviewed_template_key
    source='''class Model:
    def forward(self,x):
        return 0.5*x[:,0]
def build_model():return Model(width=16,dropout=0.1)
'''
    slots=[{'file':'train.py','function':'build_model','callee':'Model','keyword':'width','kind':'width'},
           {'file':'train.py','function':'build_model','callee':'Model','keyword':'dropout','kind':'dropout'}]
    original=reviewed_template_key({'train.py':source},slots)
    tuned=reviewed_template_key({'train.py':source.replace('width=16','width=32').replace('dropout=0.1','dropout=0.2')},slots)
    assert original[0]==tuned[0] and original[1]!=tuned[1]
    assert reviewed_template_key({'train.py':source.replace('width=16','width=1')},slots) is None
    for changed in [source.replace('0.5*x','1.0*x'),source.replace('x[:,0]','x[:,1]'),source.replace('return 0.5*x[:,0]','return x[:,0] if x.sum()>1 else x[:,1]')]:
        assert reviewed_template_key({'train.py':changed},slots)[0]!=original[0]


def test_template_full_fingerprint_keeps_parent_transition_separate(monkeypatch):
    import experiments.ontology_review_recurrent as module
    base=BASE.replace('def __init__(self):','def __init__(self,width=80):').replace('return KeywordGRU()','return KeywordGRU(width=80)')
    sources={'train.py':base};candidate={'train.py':base.replace('KeywordGRU(width=80)','KeywordGRU(width=96)')}
    slots=[{'file':'train.py','function':'build_model','callee':'KeywordGRU','keyword':'width','kind':'width'}]
    key,_=module.reviewed_template_key(sources,slots);fp,residual=module._fingerprint(module.profile(sources,KWS));assert not residual
    ref={'campaign':KWS,'source_sha256':module.source_sha(sources),'parent_source_sha256':'different-parent',
         'classification':'preserving','notes':'source reviewed','fingerprint':fp,'fingerprint_complete':True,
         'evidence':[{'file':'train.py','code':base}], 'family_signature':{'recurrence':'GRU'},
         'reviewed_template':{'numeric_slots':slots,'whole_source_ast_sha256':key}}
    monkeypatch.setattr(module,'_direct_references',lambda:[ref]);monkeypatch.setattr(module,'_automatic_pair',lambda *a:None)
    result=module.review_pair({'train.py':BASE},candidate,KWS)
    assert result['classification']=='uncertain' and result['fingerprint_complete']
    assert result['settings']['reviewed_numeric_slots']['build_model.KeywordGRU.width']==96
    assert result['evidence'][0]['code']==candidate['train.py']
    same_template=module.review_pair(sources,candidate,KWS)
    assert same_template['classification']=='preserving' and same_template['fingerprint_complete']
    changed={'train.py':candidate['train.py'].replace('total+output','total+output.square()')}
    assert module.review_pair(sources,changed,KWS) is None


def test_literal_module_list_index_is_resolved_without_skipping_unknown_elements():
    before=BASE.replace('self.gru=nn.GRU(20,80,batch_first=True)','self.grus=nn.ModuleList([nn.GRU(20,80,batch_first=True)])').replace('self.gru(', 'self.grus[0](')
    p=profile({'train.py':before},KWS)
    assert 'recurrent_step' not in p['errors']
    assert p['mechanism']['recurrent_operator']==['GRU:forward']
    missing=before.replace('[nn.GRU(20,80,batch_first=True)]','[unknown_module,nn.GRU(20,80,batch_first=True)]')
    p=profile({'train.py':missing},KWS)
    assert 'recurrent_step' in p['errors']
    bidirectional=before.replace('batch_first=True)','batch_first=True,bidirectional=True)')
    p=profile({'train.py':bidirectional},KWS)
    assert p['mechanism']['recurrent_operator']==['GRU:bidirectional']


def test_helper_callable_argument_retains_known_module_identity():
    import ast
    from experiments.ontology_review_recurrent import Equation
    helper=ast.parse('def apply(self,projection,x):\n    return projection(x)').body[0]
    forward=ast.parse('def forward(self,x):\n    return self.apply(self.head,x)').body[0]
    equation=Equation({'self.head':{'kind':'Linear'}},{'apply':helper},{})
    assert equation.run(forward)==('affine',('signal',0))


def test_coordinated_learned_latent_split_widths_preserve_all_other_source():
    from experiments.ontology_recurrent_readout import same_latent_split_widths
    before='''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.down=nn.Linear(8,9)
        self.up=nn.ModuleList([nn.Linear(3,8),nn.Linear(3,8),nn.Linear(3,8)])
    def forward(self,x):
        r,z,n=torch.split(torch.tanh(self.down(x)),(3,3,3),dim=1)
        reset=self.up[0](r)
        update=self.up[1](z)
        candidate=self.up[2](n)
        return torch.sigmoid(reset)*(1.0-torch.sigmoid(update))+torch.tanh(candidate)
def build_model():return Model()
'''
    after=before.replace('Linear(8,9)','Linear(8,8)').replace('ModuleList([nn.Linear(3,8)','ModuleList([nn.Linear(2,8)').replace('(3,3,3),dim=1','(2,3,3),dim=1')
    assert same_latent_split_widths({'train.py':before},{'train.py':after})
    for wrong in [after.replace('Linear(8,8)','Linear(8,9)'),after.replace('ModuleList([nn.Linear(2,8)','ModuleList([nn.Linear(3,8)'),after.replace('dim=1','dim=0'),after.replace('1.0-torch','2.0-torch'),after.replace('torch.tanh(self.down(x)),','x,'),after.replace('self.up[0](r)','self.up[0](r.square())')]:
        assert not same_latent_split_widths({'train.py':before},{'train.py':wrong})


def test_functional_affine_and_normalization_do_not_hide_signal_generated_weights():
    import ast,pytest
    from experiments.ontology_review_recurrent import Equation,Unsupported
    equation=Equation({}, {}, {});equation.values={'x':('signal',0)}
    assert equation.expr(ast.parse('F.linear(x,self.weight,self.bias)',mode='eval').body)==('affine',('signal',0))
    assert equation.expr(ast.parse('F.layer_norm(x,(20,),None,None)',mode='eval').body)==('signal',0)
    for expression in ['F.linear(x,x[0])','F.linear(x,weight=x[0])','F.layer_norm(x,(20,),x[0])','F.layer_norm(x,(20,),weight=x[0])','F.layer_norm(x,(20,),eps=x.sum())']:
        with pytest.raises(Unsupported):equation.expr(ast.parse(expression,mode='eval').body)


def test_fresh_local_masked_write_keeps_selector_and_replacement_dataflow():
    import ast
    from experiments.ontology_review_recurrent import Equation,_walk
    method=ast.parse('''def forward(self,x):
    indices=x.square().sum(dim=1).topk(2).indices
    candidates=torch.zeros(x.shape[0],dtype=torch.bool)
    candidates[indices]=True
    return candidates
''').body[0]
    result=Equation({}, {}, {}).run(method)
    assert result[0]=='indexed-update'
    assert any(isinstance(n,tuple) and n and n[0]=='ranked-membership' for n in _walk(result))


def test_mutable_tensor_aliases_and_external_writes_remain_unhandled():
    import ast,pytest
    from experiments.ontology_review_recurrent import Equation,Unsupported
    template='''def forward(self,x):
    logits=x.clone()
    INSERT
    logits[0]=1.0
    return logits
'''
    for statement in ['alias=logits','alias=(logits,)','alias=logits.view(-1)','self.helper(logits)']:
        with pytest.raises(Unsupported):Equation({}, {}, {}).run(ast.parse(template.replace('INSERT',statement)).body[0])
    with pytest.raises(Unsupported):Equation({}, {}, {}).run(ast.parse('def forward(self,x):\n    x[0]=1\n    return x').body[0])


def test_exit_policy_can_read_output_head_parameters_without_constructing_weights():
    exit_hook='''    def exit_mask(self,state,logits,step,total_steps):
        margin=logits.amax(dim=1)-logits.amin(dim=1)
        bound=self.head.weight.abs().sum()
        return margin>bound
'''
    source=BASE.replace('    def frame_schedule',exit_hook+'    def frame_schedule')
    answer=review(source,source)
    assert answer['fingerprint_complete']
    assert 'exit rule reads' in answer['fingerprint']['parameter_construction']
    changed=source.replace('bound=self.head.weight.abs().sum()','self.head.weight.copy_(self.head.weight.square())\n        bound=self.head.weight.abs().sum()')
    answer=review(changed,changed)
    assert not answer['fingerprint_complete']


def test_parameter_role_and_scatter_value_remain_explicit():
    import ast
    from experiments.ontology_review_recurrent import Equation
    equation=Equation({}, {}, {});equation.values={'x':('signal',0),'index':('signal',1)}
    assert equation.expr(ast.parse('self.head.weight',mode='eval').body)!=equation.expr(ast.parse('self.head.bias',mode='eval').body)
    assert equation.expr(ast.parse('x.scatter(1,index,0.0)',mode='eval').body)!=equation.expr(ast.parse('x.scatter(1,index,1.0)',mode='eval').body)


def test_order_statistics_keep_direction_and_canonicalize_explicit_defaults():
    import ast
    from experiments.ontology_review_recurrent import Equation
    equation=Equation({}, {}, {});equation.values={'x':('signal',0)}
    evaluate=lambda source:equation.expr(ast.parse(source,mode='eval').body)
    assert evaluate('x.topk(2)')==evaluate('x.topk(2,dim=-1,largest=True,sorted=True)')
    assert evaluate('x.topk(2)')==evaluate('torch.topk(x,2)')
    assert evaluate('x.topk(2)')!=evaluate('x.topk(2,largest=False)')
    assert evaluate('x.sort()')==evaluate('x.sort(dim=-1,descending=False,stable=False)')
    assert evaluate('x.sort()')!=evaluate('x.sort(descending=True)')


def test_boolean_ranked_membership_ignores_output_order_but_keeps_direction():
    import ast
    from experiments.ontology_review_recurrent import Equation
    template='''def forward(self,x):
    indices=x.square().sum(dim=1).topk(2,largest=LARGEST,sorted=SORTED).indices
    mask=torch.zeros(x.shape[0],dtype=torch.bool)
    mask[indices]=True
    return mask
'''
    def evaluate(largest,ordered):
        source=template.replace('LARGEST',str(largest)).replace('SORTED',str(ordered))
        return Equation({}, {}, {}).run(ast.parse(source).body[0])
    assert evaluate(True,True)==evaluate(True,False)
    assert evaluate(True,True)!=evaluate(False,True)


def test_boolean_reductions_keep_axes_and_reject_unaccounted_output_mutation():
    import ast,pytest
    from experiments.ontology_review_recurrent import Equation,Unsupported
    equation=Equation({}, {}, {});equation.values={'x':('signal',0)}
    evaluate=lambda source:equation.expr(ast.parse(source,mode='eval').body)
    assert evaluate('x.any()')!=evaluate('x.all()')
    assert evaluate('x.any(dim=0)')!=evaluate('x.any(dim=1)')
    with pytest.raises(Unsupported):evaluate('x.any(out=x)')


def test_energy_routing_direct_review_keeps_mean_history_separate():
    from experiments.ontology_review_recurrent import _direct_references
    ref=next(r for r in _direct_references() if r['campaign']==KWS and r['run_id'].endswith('b02-c0') and r['proposal']==35)
    assert ref['fingerprint_complete'] and ref['classification']=='changing'
    assert set(ref['changed_components'])=={'routing','conditional_compute','exit_policy'}
    assert 'mean' in ref['fingerprint']['readout_history']
    assert 'batch' in ref['fingerprint']['conditional_compute']


def test_fixed_contrast_layer_to_buffer_direct_review_is_preserving():
    from pathlib import Path
    from experiments.ontology_review_recurrent import review_pair
    root=_fixture_repository_root()/'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b02-c0/candidates'
    before=next(root.glob('886e9bd17a2e*'))
    after=next(root.glob('d3c4f38df472*'))
    read=lambda path:{file.relative_to(path).as_posix():file.read_text(encoding='utf-8-sig') for file in path.rglob('*.py')}
    result=review_pair(read(before),read(after),KWS)
    assert result['classification']=='preserving'
    assert result['fingerprint_complete']
    assert result['changed_components']==[]
    assert 'fixed seven-to-eight' in result['fingerprint']['feedforward']


def test_certificate_slack_tail_exit_direct_review_is_changing():
    from pathlib import Path
    from experiments.ontology_review_recurrent import review_pair
    root=_fixture_repository_root()/'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b02-c0/candidates'
    before=next(root.glob('d3c4f38df472*'))
    after=next(root.glob('24339b7ce11c*'))
    read=lambda path:{file.relative_to(path).as_posix():file.read_text(encoding='utf-8-sig') for file in path.rglob('*.py')}
    result=review_pair(read(before),read(after),KWS)
    assert result['classification']=='changing'
    assert result['fingerprint_complete']
    assert set(result['changed_components'])=={'routing','conditional_compute','exit_policy'}
    assert 'heuristic exit' in result['fingerprint']['exit_policy']


def test_literal_sequential_child_index_is_traced_but_dynamic_index_is_not():
    import ast
    import pytest
    from experiments.ontology_review_recurrent import Equation, Unsupported
    modules={'self.head[0]':{'kind':'Linear'}}
    methods={}
    fixed=Equation(modules,methods,{})
    fixed.values['self.head[0]']=('module-reference','self.head[0]')
    fixed.values['x']=('signal',0)
    assert fixed.expr(ast.parse('self.head[0](x)',mode='eval').body)==('affine',('signal',0))
    dynamic=Equation(modules,methods,{})
    dynamic.values['index']=('signal',1)
    dynamic.values['x']=('signal',0)
    with pytest.raises(Unsupported):
        dynamic.expr(ast.parse('self.head[index](x)',mode='eval').body)


def test_lagged_cross_covariance_har_direct_review_is_changing():
    from pathlib import Path
    from experiments.ontology_review_recurrent import review_pair
    root=_fixture_repository_root()/'data/c0c3/uci-har-pareto-v21/runs/uci-har-pareto-v21-uci-har-pareto-openevolve-b01-c1/candidates'
    before=next(root.glob('675286c44a8b*'))
    after=next(root.glob('a72f3b1eaf18*'))
    read=lambda path:{file.relative_to(path).as_posix():file.read_text(encoding='utf-8-sig') for file in path.rglob('*.py')}
    result=review_pair(read(before),read(after),HAR)
    assert result['classification']=='changing'
    assert result['fingerprint_complete']
    assert set(result['changed_components'])=={'sensor_fusion','connectivity','mixing','aggregation','temporal_readout'}
    assert 'lags 0, 2 and 8' in result['fingerprint']['aggregation']
    assert result['settings']['magnitude_cadence_lags']==[8,32]


def test_lagged_cross_covariance_zero_lag_compression_is_exact_pair_bound():
    """P81 removes only duplicate entries from its zero-lag covariance."""
    from pathlib import Path
    from experiments.ontology_review_recurrent import review_pair, source_sha

    root = _fixture_repository_root() / (
        'data/c0c3/uci-har-pareto-v21/runs/'
        'uci-har-pareto-v21-uci-har-pareto-openevolve-b01-c1/candidates'
    )
    before = next(root.glob('a72f3b1eaf18*'))
    after = next(root.glob('579df9e1c2a9*'))
    read = lambda path: {
        file.relative_to(path).as_posix(): file.read_text(encoding='utf-8-sig')
        for file in path.rglob('*.py')
    }
    before_source, after_source = read(before), read(after)
    assert source_sha(before_source) == '2f1b748cf9d61ee6ec87e1b9e8c544b2906f2d017d4677ac469a83da5dbb2e73'
    assert source_sha(after_source) == '0049ae64c0d4237138f7d2d7d34ac24bf6d461ce4a58529afc501a5f9bafd074'

    result = review_pair(before_source, after_source, HAR)
    assert result['classification'] == 'preserving'
    assert result['fingerprint_complete']
    assert result['residual_components'] == []
    assert result['changed_components'] == []
    assert result['reviewer'] == 'har-lagged-cross-covariance-zero-lag-exact-pair-v1'
    assert result['settings']['zero_lag_coordinates'] == {
        'before': 225, 'after': 120, 'removed_duplicate_off_diagonals': 105,
    }
    assert 'upper-triangular zero-lag covariance' in result['fingerprint']['aggregation']

    changed = dict(after_source)
    changed['train.py'] = changed['train.py'].replace(
        'nn.Linear(classifierInput, 30)', 'nn.Linear(classifierInput, 31)', 1
    )
    from experiments.ontology_har_lagged_covariance import review_pair as exact_pair_review
    assert exact_pair_review(before_source, changed) is None


def test_raw_lag_hidden_width_sweep_is_hash_bound_and_complete():
    """Eleven RawLag width steps are exact pairs; the attention branch is not."""
    from pathlib import Path
    from experiments.ontology_har_raw_lag_width_sweep import EXACT_PAIRS, review_pair as exact_pair_review
    from experiments.ontology_review_recurrent import review_pair, source_sha

    root = _fixture_repository_root() / (
        'data/c0c3/uci-har-pareto-v21/runs/'
        'uci-har-pareto-v21-uci-har-pareto-openevolve-b01-c1/candidates'
    )
    source_for_sha = {}
    for directory in root.iterdir():
        if directory.is_dir():
            source = {
                file.relative_to(directory).as_posix(): file.read_text(encoding='utf-8-sig')
                for file in directory.rglob('*.py')
            }
            source_for_sha[source_sha(source)] = source

    assert [entry[0] for entry in EXACT_PAIRS] == [158, 159, *range(161, 170)]
    for proposal, before_sha, after_sha, before_width, after_width in EXACT_PAIRS:
        before_source, after_source = source_for_sha[before_sha], source_for_sha[after_sha]
        result = review_pair(before_source, after_source, HAR)
        assert result['classification'] == 'preserving'
        assert result['fingerprint_complete']
        assert result['residual_components'] == []
        assert result['changed_components'] == []
        assert result['reviewer'] == 'har-raw-lag-hidden-width-sweep-exact-pairs-v1'
        assert result['settings']['proposal'] == proposal
        assert result['settings']['hidden_width'] == {'before': before_width, 'after': after_width}
        assert result['settings']['changed_source_components'] == [
            'build_model RawLagInteractionNet hiddenWidth keyword',
        ]
        assert f'{after_width}-wide affine' in result['fingerprint']['feedforward']

    changed = dict(source_for_sha[EXACT_PAIRS[0][2]])
    changed['train.py'] = changed['train.py'].replace('hiddenWidth=20,', 'hiddenWidth=99,', 1)
    assert exact_pair_review(source_for_sha[EXACT_PAIRS[0][1]], changed) is None


def test_har_b05_c2_deployment_compression_ledger_is_hash_bound_and_complete():
    """All 79 B05-C2 compression steps require their recorded SHA pair."""
    from pathlib import Path
    from experiments.ontology_har_deployment_compression import EXACT_PAIRS, review_pair as exact_pair_review
    from experiments.ontology_review_recurrent import review_pair, source_sha

    root = _fixture_repository_root() / (
        'data/c0c3/uci-har-pareto-v21/runs/'
        'uci-har-pareto-v21-uci-har-pareto-openevolve-b05-c2/candidates'
    )
    source_for_sha = {}
    for directory in root.iterdir():
        if directory.is_dir():
            source = {
                file.relative_to(directory).as_posix(): file.read_text(encoding='utf-8-sig')
                for file in directory.rglob('*.py')
            }
            source_for_sha[source_sha(source)] = source

    assert [entry['proposal'] for entry in EXACT_PAIRS] == list(range(122, 201))
    for entry in EXACT_PAIRS:
        before_source = source_for_sha[entry['parent_sha256']]
        after_source = source_for_sha[entry['candidate_sha256']]
        result = review_pair(before_source, after_source, HAR)
        assert result['classification'] == 'preserving'
        assert result['fingerprint_complete']
        assert result['residual_components'] == []
        assert result['changed_components'] == []
        assert result['reviewer'] == 'har-b05-c2-deployment-compression-exact-pairs-v1'
        assert result['settings']['proposal'] == entry['proposal']
        assert result['settings']['deployment_omissions'] == {
            'before': entry['before_omitted'], 'after': entry['after_omitted'],
        }
        assert result['settings']['deployment_main_head_input'] == {
            'before': entry['parent_head_input'], 'after': entry['candidate_head_input'],
        }
        assert f"{entry['candidate_head_input']}-input frozen relative-logit head" in result['fingerprint']['routing']

    first = EXACT_PAIRS[0]
    changed = dict(source_for_sha[first['candidate_sha256']])
    changed['train.py'] = changed['train.py'].replace('for _ in range(10)', 'for _ in range(99)', 1)
    assert exact_pair_review(source_for_sha[first['parent_sha256']], changed) is None

    wrong_parent = dict(source_for_sha[first['parent_sha256']])
    wrong_parent['train.py'] = wrong_parent['train.py'].replace('for _ in range(6)', 'for _ in range(99)', 1)
    assert exact_pair_review(wrong_parent, source_for_sha[first['candidate_sha256']]) is None


def test_har_b04_c2_inference_readout_pruning_ledger_is_hash_bound_and_complete():
    """59 exact B04-C2 pairs exclude the adjacent pooling transition."""
    from pathlib import Path
    from experiments.ontology_har_inference_readout_pruning import EXACT_PAIRS, review_pair as exact_pair_review
    from experiments.ontology_review_recurrent import review_pair, source_sha

    root = _fixture_repository_root() / (
        'data/c0c3/uci-har-pareto-v21/runs/'
        'uci-har-pareto-v21-uci-har-pareto-openevolve-b04-c2/candidates'
    )
    source_for_sha = {}
    for directory in root.iterdir():
        if directory.is_dir():
            source = {file.relative_to(directory).as_posix(): file.read_text(encoding='utf-8-sig') for file in directory.rglob('*.py')}
            source_for_sha[source_sha(source)] = source

    assert len(EXACT_PAIRS) == 59
    assert [entry['proposal'] for entry in EXACT_PAIRS] == [*range(135, 144), *range(145, 195)]
    for entry in EXACT_PAIRS:
        result = review_pair(source_for_sha[entry['parent_sha256']], source_for_sha[entry['candidate_sha256']], HAR)
        assert result['classification'] == 'preserving'
        assert result['fingerprint_complete'] and result['residual_components'] == [] and result['changed_components'] == []
        assert result['reviewer'] == 'har-b04-c2-inference-readout-pruning-exact-pairs-v1'
        assert result['settings']['inference_readout_prune_count'] == {'before': entry['before_prune_count'], 'after': entry['after_prune_count']}
        assert result['settings']['compact_readout_width'] == {'before': entry['parent_compact_width'], 'after': entry['candidate_compact_width']}

    first = EXACT_PAIRS[0]
    changed = dict(source_for_sha[first['candidate_sha256']])
    changed['train.py'] = changed['train.py'].replace('inferenceReadoutPruneCount = 9', 'inferenceReadoutPruneCount = 99', 1)
    assert exact_pair_review(source_for_sha[first['parent_sha256']], changed) is None


def test_har_b03_c2_compact_reset_projection_ledger_is_hash_bound_and_complete():
    from pathlib import Path
    from experiments.ontology_har_compact_reset_projection import EXACT_PAIRS, review_pair as exact_pair_review
    from experiments.ontology_review_recurrent import review_pair, source_sha
    root = _fixture_repository_root() / ('data/c0c3/uci-har-pareto-v21/runs/' 'uci-har-pareto-v21-uci-har-pareto-openevolve-b03-c2/candidates')
    source_for_sha = {}
    for directory in root.iterdir():
        if directory.is_dir():
            source = {file.relative_to(directory).as_posix(): file.read_text(encoding='utf-8-sig') for file in directory.rglob('*.py')}
            source_for_sha[source_sha(source)] = source
    assert [x['proposal'] for x in EXACT_PAIRS] == [53,54,55,56,61,62,63,64,65,66]
    for entry in EXACT_PAIRS:
        result = review_pair(source_for_sha[entry['parent_sha256']], source_for_sha[entry['candidate_sha256']], HAR)
        assert result['classification'] == 'preserving' and result['fingerprint_complete']
        assert result['residual_components'] == [] and result['changed_components'] == []
        assert result['reviewer'] == 'har-b03-c2-compact-reset-projection-exact-pairs-v1'
        assert result['settings']['third_omission_transitions'] == {'before':entry['before_transitions'],'after':entry['after_transitions']}
    first = EXACT_PAIRS[0]
    changed = dict(source_for_sha[first['candidate_sha256']])
    changed['train.py'] = changed['train.py'].replace('torch.arange(9, dtype=torch.long)', 'torch.arange(99, dtype=torch.long)', 1)
    assert exact_pair_review(source_for_sha[first['parent_sha256']], changed) is None


def test_har_b04_c1_context_pooling_ledger_is_hash_bound_and_complete():
    from pathlib import Path
    from experiments.ontology_har_context_pooling import EXACT_PAIRS, review_pair as exact_pair_review
    from experiments.ontology_review_recurrent import review_pair, source_sha
    root = _fixture_repository_root() / ('data/c0c3/uci-har-pareto-v21/runs/' 'uci-har-pareto-v21-uci-har-pareto-openevolve-b04-c1/candidates')
    source_for_sha = {}
    for directory in root.iterdir():
        if directory.is_dir():
            source = {file.relative_to(directory).as_posix(): file.read_text(encoding='utf-8-sig') for file in directory.rglob('*.py')}
            source_for_sha[source_sha(source)] = source
    assert [x['proposal'] for x in EXACT_PAIRS] == list(range(52,60))
    for entry in EXACT_PAIRS:
        result = review_pair(source_for_sha[entry['parent_sha256']], source_for_sha[entry['candidate_sha256']], HAR)
        assert result['classification'] == 'preserving' and result['fingerprint_complete']
        assert result['residual_components'] == [] and result['changed_components'] == []
        assert result['settings']['adaptive_context_tokens'] == {'before':entry['before_context_tokens'],'after':entry['after_context_tokens']}
        assert result['reviewer'] == 'har-b04-c1-context-pooling-exact-pairs-v1'
    first = EXACT_PAIRS[0]; changed = dict(source_for_sha[first['candidate_sha256']])
    changed['train.py'] = changed['train.py'].replace('adaptive_avg_pool1d(x, 18)', 'adaptive_avg_pool1d(x, 99)', 1)
    assert exact_pair_review(source_for_sha[first['parent_sha256']], changed) is None


def test_har_b03_c2_fourth_reset_omission_ledger_is_hash_bound_and_complete():
    from pathlib import Path
    from experiments.ontology_har_fourth_reset_omission import EXACT_PAIRS, review_pair as exact_pair_review
    from experiments.ontology_review_recurrent import review_pair, source_sha
    root = _fixture_repository_root() / ('data/c0c3/uci-har-pareto-v21/runs/' 'uci-har-pareto-v21-uci-har-pareto-openevolve-b03-c2/candidates')
    source_for_sha = {}
    for directory in root.iterdir():
        if directory.is_dir():
            source = {file.relative_to(directory).as_posix(): file.read_text(encoding='utf-8-sig') for file in directory.rglob('*.py')}
            source_for_sha[source_sha(source)] = source
    assert [x['proposal'] for x in EXACT_PAIRS] == list(range(72,77))
    for entry in EXACT_PAIRS:
        result = review_pair(source_for_sha[entry['parent_sha256']], source_for_sha[entry['candidate_sha256']], HAR)
        assert result['classification'] == 'preserving' and result['fingerprint_complete']
        assert result['residual_components'] == [] and result['changed_components'] == []
        assert result['settings']['fourth_omission_transitions'] == {'before':entry['before_fourth_omissions'],'after':entry['after_fourth_omissions']}
        assert result['settings']['fourth_projection_input'] == 24
    first = EXACT_PAIRS[0]; changed = dict(source_for_sha[first['candidate_sha256']])
    changed['train.py'] = changed['train.py'].replace('torch.arange(7, dtype=torch.long)', 'torch.arange(99, dtype=torch.long)', 1)
    assert exact_pair_review(source_for_sha[first['parent_sha256']], changed) is None
    wrong_parent = dict(source_for_sha[first['parent_sha256']])
    wrong_parent['train.py'] = wrong_parent['train.py'].replace('torch.arange(3, dtype=torch.long)', 'torch.arange(98, dtype=torch.long)', 1)
    assert exact_pair_review(wrong_parent, source_for_sha[first['candidate_sha256']]) is None


def test_har_family_signature_keeps_structural_module_multiplicity_and_call_order():
    """Direct B01-C0 regressions for previously collapsed HAR families."""
    from pathlib import Path
    from experiments.ontology_review_recurrent import review_pair, source_sha
    root=_fixture_repository_root()/'data/c0c3/uci-har-pareto-v21/runs/uci-har-pareto-v21-uci-har-pareto-openevolve-b01-c0/candidates'
    read=lambda path:{file.relative_to(path).as_posix():file.read_text(encoding='utf-8-sig') for file in path.rglob('*.py')}
    cases=[
        # Dense five-tap stem -> depthwise five-tap stem plus dense mix.
        ('47e4e990a821','79e315194404','0cd243021a5cf79b7553507f4b99b041f71113c895153066ce0c3983bf7207c8','386599baa65f88ca461abdcff0cd55bd2910d65d3ffcadc53968724bdd7abc54','temporal_operator'),
        # Removing the pre-mixing ReLU changes the ordered operator graph.
        ('1e52c936926d','2c2e373ca2b4','adb63ddd9ff813451cc5e94781c769d312d61c2fc045f4fa06f42c3926ad2b62','127ed6d23384b76c9561f4b50b57bfe7feb794a7560bc9e15109ffbb06b14116','connectivity'),
        # 31 -> 32 retained temporal positions changes the pool schedule.
        ('6d2b6962d2a4','2c9a2cfb5e25','1741078787021b182a55a2ee767c12b136fb01a3361cdcbe4b62869bdc0b6754','f36cfcbf96fe7cbc6401cd66020da02c843672b167eee6636aa4f89b5b0d25ec','temporal_operator'),
    ]
    for before_prefix,after_prefix,before_sha,after_sha,component in cases:
        before,after=next(root.glob(before_prefix+'*')),next(root.glob(after_prefix+'*'))
        before_source,after_source=read(before),read(after)
        assert source_sha(before_source)==before_sha
        assert source_sha(after_source)==after_sha
        result=review_pair(before_source,after_source,HAR)
        assert result['classification']=='changing'
        assert component in result['changed_components']


def test_fixed_reference_logit_quotient_is_source_bound_and_loss_preserving():
    """P124 removes only the softmax's common-logit coordinate.

    This reads the captured sources; it never imports or executes campaign code.
    """
    import hashlib
    import torch
    from torch.nn import functional as F

    root = _fixture_repository_root() / (
        'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/'
        'tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b01-c0/candidates'
    )
    before = next(root.glob('85f26b80d8ad*')) / 'train.py'
    after = next(root.glob('fd3255a3d6b3*')) / 'train.py'
    assert hashlib.sha256(before.read_bytes()).hexdigest() == (
        'cc177289b44b49d7e73202b6dd9e0e2acce87211b8cd9a342d089e5f575c17e9'
    )
    assert hashlib.sha256(after.read_bytes()).hexdigest() == (
        '12a5d204d032e0f73934de3595735c47f3297051e9724f79ef033acbb8c4d1e5'
    )
    source = after.read_text(encoding='utf-8-sig')
    assert 'self.classifier = nn.Linear(1065, 7)' in source
    assert 'torch.zeros_like(contrast_logits[:, :1])' in source
    assert 'return F.cross_entropy(logits, labels, label_smoothing=0.03)' in source

    logits = torch.tensor(
        [[1.3, -0.2, 0.7, 2.1, -3.0, 0.4, 0.0, 1.2],
         [-1.0, 4.0, 0.1, -0.5, 2.0, 3.1, -2.0, 0.6]]
    )
    quotient = torch.cat(
        (logits[:, :7] - logits[:, 7:], torch.zeros_like(logits[:, :1])),
        dim=1,
    )
    labels = torch.tensor([3, 1])
    assert torch.equal(logits.argmax(dim=1), quotient.argmax(dim=1))
    assert torch.allclose(
        F.cross_entropy(logits, labels, label_smoothing=0.03),
        F.cross_entropy(quotient, labels, label_smoothing=0.03),
    )


def _kws_b01_c2_confidence_sources(before_prefix, after_prefix):
    root = _fixture_repository_root() / (
        'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/'
        'tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b01-c2/candidates'
    )
    read = lambda path: {
        file.relative_to(path).as_posix(): file.read_text(encoding='utf-8-sig')
        for file in path.rglob('*.py')
    }
    return read(next(root.glob(before_prefix + '*'))), read(next(root.glob(after_prefix + '*')))


def test_kws_b01_c2_confidence_threshold_ledger_is_complete_and_source_bound():
    """All retained B01-C2 threshold edits keep one exact exit topology."""
    from experiments.ontology_review_recurrent import source_sha

    shared_fingerprint = None
    for _, before_prefix, after_prefix, before_sha, after_sha in KWS_B01_C2_CONFIDENCE_CASES:
        before_source, after_source = _kws_b01_c2_confidence_sources(before_prefix, after_prefix)
        assert source_sha(before_source) == before_sha
        assert source_sha(after_source) == after_sha
        assert before_source['train.py'].count('confidence >=') == 1
        assert after_source['train.py'].count('confidence >=') == 1

        result = review_pair(before_source, after_source, KWS)
        assert result['classification'] == 'preserving'
        assert result['fingerprint_complete'] is True
        assert result['residual_components'] == []
        assert result['changed_components'] == []
        assert result['reviewer'] == 'recurrent-kws-b01-c2-direct-source-v1'
        assert result['evidence'][0]['line'] == 152
        assert 'confidence >=' in result['evidence'][0]['code']
        if shared_fingerprint is None:
            shared_fingerprint = result['fingerprint']
        else:
            assert result['fingerprint'] == shared_fingerprint


@pytest.mark.parametrize(
    'label,before_prefix,after_prefix,before_sha,after_sha',
    KWS_B01_C2_CONFIDENCE_CASES,
    ids=[case[0] for case in KWS_B01_C2_CONFIDENCE_CASES],
)
def test_kws_b01_c2_confidence_threshold_ledger_rejects_parent_or_exit_mutation(
    label, before_prefix, after_prefix, before_sha, after_sha,
):
    """A matched child may not inherit a confidence decision from another source."""
    from experiments.ontology_review_recurrent import (
        _kws_b01_c2_source_review, source_sha,
    )

    before_source, after_source = _kws_b01_c2_confidence_sources(before_prefix, after_prefix)
    altered_parent = dict(before_source)
    altered_parent['train.py'] = altered_parent['train.py'].replace(
        'BATCH_SIZE = 128', 'BATCH_SIZE = 129', 1,
    )
    altered_child = dict(after_source)
    altered_child['train.py'] = altered_child['train.py'].replace(
        'confidence >=', 'confidence >', 1,
    )
    assert source_sha(altered_parent) != before_sha
    assert source_sha(altered_child) != after_sha
    for parent, child in ((altered_parent, after_source), (before_source, altered_child)):
        assert _kws_b01_c2_source_review(parent, child, KWS) is None


def test_retained_kws_b01_c2_cache_and_early_exit_reviews_are_source_bound():
    """Private auxiliary-loss caches must not hide deployed exit changes.

    These are retained source bundles, parsed only.  The table exercises the
    narrow cache extension as well as both preserving readout/projection cases
    and the new antepenultimate termination path.
    """
    from pathlib import Path
    from experiments.ontology_review_recurrent import CORE, TASK_KEYS, source_sha

    root = _fixture_repository_root() / (
        'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/'
        'tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b01-c2/candidates'
    )
    cases = [
        ('34d2cd869dc0', '3c5edd494270', 'preserving'),
        ('e75a8f47dcdb', 'bcd656311f64', 'preserving'),
        ('5d7fe142ec71', '76f00a0237f1', 'preserving'),
        ('e75a8f47dcdb', '3110c566a3ad', 'preserving'),
        ('e75a8f47dcdb', '39c03065e0a2', 'preserving'),
        ('556c70a14683', '9341afad6921', 'preserving'),
        ('64d11793c6a9', '6246dbec7e13', 'preserving'),
        ('c8de012a21d8', '15bcf1a71aca', 'changing'),
        ('81ae21c12d53', 'b3f43cc0a206', 'changing'),
        ('128054fcef64', 'af96070657a4', 'changing'),
        ('81ae21c12d53', 'd39d68bd2956', 'changing'),
        ('81ae21c12d53', '63c63b6e5d72', 'changing'),
        ('c8de012a21d8', 'ab8c795d6de9', 'changing'),
        ('81ae21c12d53', 'c0a43655692f', 'changing'),
        ('128054fcef64', 'f37057621cf9', 'preserving'),
        ('4481f49b5ca5', 'e885c716689e', 'preserving'),
        ('c8de012a21d8', '65995b0f34d6', 'preserving'),
        ('81ae21c12d53', 'f102dbd46689', 'preserving'),
        ('128054fcef64', 'fc5c05f73ce4', 'preserving'),
        ('4481f49b5ca5', 'e3744cefdfb0', 'preserving'),
    ]
    read = lambda path: {
        file.relative_to(path).as_posix(): file.read_text(encoding='utf-8-sig')
        for file in path.rglob('*.py')
    }
    required = set(CORE + TASK_KEYS['kws'].split())
    for before_prefix, after_prefix, classification in cases:
        before, after = next(root.glob(before_prefix + '*')), next(root.glob(after_prefix + '*'))
        before_source, after_source = read(before), read(after)
        result = review_pair(before_source, after_source, KWS)
        assert result['classification'] == classification
        assert result['fingerprint_complete']
        assert result['residual_components'] == []
        assert required <= set(result['fingerprint'])
        assert result['before_source_sha256'] == source_sha(before_source)
        assert result['source_sha256'] == source_sha(after_source)


def test_retained_kws_specialist_reranker_is_changing_but_keeps_its_mutation_residual():
    """A positive source witness may classify a bounded-parser residual."""
    from pathlib import Path

    root = _fixture_repository_root() / (
        'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/'
        'tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b01-c2/candidates'
    )
    before, after = next(root.glob('bd0530874092*')), next(root.glob('3e84857a4faa*'))
    read = lambda path: {
        file.relative_to(path).as_posix(): file.read_text(encoding='utf-8-sig')
        for file in path.rglob('*.py')
    }
    result = review_pair(read(before), read(after), KWS)
    assert result['classification'] == 'changing'
    assert not result['fingerprint_complete']
    assert result['residual_components'] == ['classify']
    assert set(result['changed_components']) == {'routing', 'conditional_compute', 'output'}


def test_kws_b05_c0_progressive_deployment_compression_ledger_is_hash_bound_and_complete():
    """P60-P66 add exactly one covariance-selected deployment drop per step."""
    from pathlib import Path
    from experiments.ontology_kws_progressive_deployment_compression import EXACT_PAIRS
    from experiments.ontology_review_recurrent import source_sha

    root = _fixture_repository_root() / (
        'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/'
        'tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b05-c0/candidates'
    )
    read = lambda path: {
        file.relative_to(path).as_posix(): file.read_text(encoding='utf-8-sig')
        for file in path.rglob('*.py')
    }
    sources = {}
    for directory in root.iterdir():
        if directory.is_dir():
            source = read(directory)
            sources[source_sha(source)] = source

    assert [entry[0] for entry in EXACT_PAIRS] == list(range(60, 67))
    for proposal, before, after, parent_sha, candidate_sha in EXACT_PAIRS:
        result = review_pair(sources[parent_sha], sources[candidate_sha], KWS)
        assert result['classification'] == 'changing'
        assert result['fingerprint_complete'] is True
        assert result['residual_components'] == []
        assert result['reviewer'] == 'kws-b05-c0-progressive-deployment-compression-exact-pairs-v1'
        assert result['settings']['deployment_dropped_coordinates'] == {'before': before, 'after': after}
        assert result['settings']['deployment_relative_head_input'] == {'before': 114-before, 'after': 114-after}
        assert result['before_source_sha256'] == parent_sha
        assert result['source_sha256'] == candidate_sha

    proposal, before, after, parent_sha, candidate_sha = EXACT_PAIRS[0]
    changed = dict(sources[candidate_sha])
    changed['train.py'] = changed['train.py'].replace(
        'full_classifier.in_features - 3', 'full_classifier.in_features - 99', 1,
    )
    from experiments.ontology_kws_progressive_deployment_compression import review_pair as exact_review
    assert exact_review(sources[parent_sha], changed) is None



def test_kws_b05_c0_progressive_deployment_compression_v2_ledger_is_hash_bound_and_complete():
    """P88-P121 are an immutable one-coordinate deployment compression chain."""
    from experiments.ontology_kws_progressive_deployment_compression_v2 import EXACT_PAIRS
    from experiments.ontology_review_recurrent import source_sha

    root = _fixture_repository_root() / (
        'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/'
        'tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b05-c0/candidates'
    )
    read = lambda path: {
        file.relative_to(path).as_posix(): file.read_text(encoding='utf-8-sig')
        for file in path.rglob('*.py')
    }
    sources = {
        source_sha(source): source
        for directory in root.iterdir() if directory.is_dir()
        for source in [read(directory)]
    }

    assert [entry[0] for entry in EXACT_PAIRS] == list(range(88, 122))
    for proposal, before, after, parent_sha, candidate_sha in EXACT_PAIRS:
        result = review_pair(sources[parent_sha], sources[candidate_sha], KWS)
        assert result['classification'] == 'changing'
        assert result['fingerprint_complete'] is True
        assert result['residual_components'] == []
        assert result['reviewer'] == 'kws-b05-c0-progressive-deployment-compression-v2-exact-pairs-v1'
        assert result['settings'] == {
            'proposal': proposal,
            'deployment_dropped_coordinates': {'before': before, 'after': after},
            'deployment_relative_head_input': {'before': 114-before, 'after': 114-after},
        }
        assert result['before_source_sha256'] == parent_sha
        assert result['source_sha256'] == candidate_sha

    proposal, before, after, parent_sha, candidate_sha = EXACT_PAIRS[0]
    changed = dict(sources[candidate_sha])
    changed['train.py'] = changed['train.py'].replace(
        'full_classifier.in_features - 18', 'full_classifier.in_features - 99', 1,
    )
    from experiments.ontology_kws_progressive_deployment_compression_v2 import review_pair as exact_review
    assert exact_review(sources[parent_sha], changed) is None


def test_kws_b05_c0_calibrated_deployment_compression_ledger_is_hash_bound_and_complete():
    """P127-P140 add a calibrated selector coordinate per immutable pair."""
    from experiments.ontology_kws_calibrated_deployment_compression import EXACT_PAIRS
    from experiments.ontology_review_recurrent import source_sha

    root = _fixture_repository_root() / (
        'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/'
        'tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b05-c0/candidates'
    )
    read = lambda path: {
        file.relative_to(path).as_posix(): file.read_text(encoding='utf-8-sig')
        for file in path.rglob('*.py')
    }
    sources = {
        source_sha(source): source
        for directory in root.iterdir() if directory.is_dir()
        for source in [read(directory)]
    }

    assert [entry[0] for entry in EXACT_PAIRS] == list(range(127, 141))
    for proposal, before, after, parent_sha, candidate_sha in EXACT_PAIRS:
        result = review_pair(sources[parent_sha], sources[candidate_sha], KWS)
        assert result['classification'] == 'changing'
        assert result['fingerprint_complete'] is True
        assert result['residual_components'] == []
        assert result['reviewer'] == 'kws-b05-c0-calibrated-deployment-compression-exact-pairs-v1'
        assert result['settings'] == {
            'proposal': proposal,
            'deployment_dropped_coordinates': {'before': before, 'after': after},
            'deployment_relative_head_input': {'before': 114-before, 'after': 114-after},
        }
        assert result['before_source_sha256'] == parent_sha
        assert result['source_sha256'] == candidate_sha

    proposal, before, after, parent_sha, candidate_sha = EXACT_PAIRS[0]
    changed = dict(sources[candidate_sha])
    changed['train.py'] = changed['train.py'].replace(
        'full_classifier.in_features - 52', 'full_classifier.in_features - 99', 1,
    )
    from experiments.ontology_kws_calibrated_deployment_compression import review_pair as exact_review
    assert exact_review(sources[parent_sha], changed) is None


def test_kws_b05_c0_calibrated_selector_refinement_ledger_is_hash_bound_and_complete():
    """P167-P189 are immutable calibrated deployment-selector refinements."""
    from experiments.ontology_kws_calibrated_selector_refinement import EXACT_PAIRS
    from experiments.ontology_review_recurrent import source_sha

    root = _fixture_repository_root() / (
        'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/'
        'tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b05-c0/candidates'
    )
    read = lambda path: {
        file.relative_to(path).as_posix(): file.read_text(encoding='utf-8-sig')
        for file in path.rglob('*.py')
    }
    sources = {
        source_sha(source): source
        for directory in root.iterdir() if directory.is_dir()
        for source in [read(directory)]
    }

    assert [entry[0] for entry in EXACT_PAIRS] == list(range(167, 190))
    for proposal, before, after, parent_sha, candidate_sha in EXACT_PAIRS:
        result = review_pair(sources[parent_sha], sources[candidate_sha], KWS)
        assert result['classification'] == 'changing'
        assert result['fingerprint_complete'] is True
        assert result['residual_components'] == []
        assert result['reviewer'] == 'kws-b05-c0-calibrated-selector-refinement-exact-pairs-v1'
        assert result['settings'] == {
            'proposal': proposal,
            'deployment_dropped_coordinates': {'before': before, 'after': after},
            'deployment_relative_head_input': {'before': 114-before, 'after': 114-after},
        }
        assert result['before_source_sha256'] == parent_sha
        assert result['source_sha256'] == candidate_sha

    proposal, before, after, parent_sha, candidate_sha = EXACT_PAIRS[0]
    changed = dict(sources[candidate_sha])
    changed['train.py'] = changed['train.py'].replace(
        'full_classifier.in_features - 66', 'full_classifier.in_features - 99', 1,
    )
    from experiments.ontology_kws_calibrated_selector_refinement import review_pair as exact_review
    assert exact_review(sources[parent_sha], changed) is None


def test_kws_b02_c0_canonical_exit_mask_structural_review_covers_the_full_fixture_chain_and_rejects_outer_mutation():
    """The structural family rule covers every canonical P132-P167 exit-mask pair."""
    import json
    from pathlib import Path
    from experiments.ontology_review_recurrent import source_sha
    from experiments.ontology_kws_b02_c0_exit_mask_structural import review_pair as structural_review

    root = _fixture_repository_root() / (
        'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign/runs/'
        'tiny-kws-rnn-openevolve-v2-1-cpu-tiny_kws_rnn_source_only_cpu-openevolve-b02-c0/candidates'
    )
    read = lambda path: {
        file.relative_to(path).as_posix(): file.read_text(encoding='utf-8-sig')
        for file in path.rglob('*.py')
    }
    sources = {}
    for directory in root.iterdir():
        if directory.is_dir():
            source = read(directory)
            sources[source_sha(source)] = source
    report = json.loads((Path('outputs/ontology/openevolve_v21_tiny_kws_rnn.json')).read_text(encoding='utf-8'))
    rows = {
        row['proposal']: row for row in report['rows']
        if row['run_id'].endswith('openevolve-b02-c0')
    }
    pairs = [(proposal, rows[proposal - 1], rows[proposal]) for proposal in range(132, 168)]
    assert len(pairs) == 36
    for proposal, before_row, after_row in pairs:
        assert before_row['classification'] == after_row['classification'] == 'uncertain'
        result = review_pair(sources[before_row['source_sha256']], sources[after_row['source_sha256']], KWS)
        assert result['classification'] == 'changing'
        assert result['fingerprint_complete'] is True
        assert result['residual_components'] == []
        assert result['reviewer'] == 'kws-b02-c0-canonical-exit-mask-structural-v1'
        assert result['settings']['before_exit_mask_sha256'] != result['settings']['after_exit_mask_sha256']

    before_row, after_row = rows[131], rows[132]
    mutated = dict(sources[after_row['source_sha256']])
    mutated['train.py'] = mutated['train.py'].replace('nn.GRU(16, 98', 'nn.GRU(16, 97', 1)
    assert structural_review(sources[before_row['source_sha256']], mutated) is None



def test_har_b03_c2_quaternary_reset_omission_ledger_is_hash_bound_and_complete():
    from pathlib import Path
    from experiments.ontology_har_quaternary_reset_omission import EXACT_PAIRS, review_pair as exact_pair_review
    from experiments.ontology_review_recurrent import review_pair, source_sha
    root = _fixture_repository_root() / ('data/c0c3/uci-har-pareto-v21/runs/' 'uci-har-pareto-v21-uci-har-pareto-openevolve-b03-c2/candidates')
    source_for_sha = {}
    for directory in root.iterdir():
        if directory.is_dir():
            source = {file.relative_to(directory).as_posix(): file.read_text(encoding='utf-8-sig') for file in directory.rglob('*.py')}
            source_for_sha[source_sha(source)] = source
    assert [entry['proposal'] for entry in EXACT_PAIRS] == [131, 132, 134, 135, 136, 138]
    for entry in EXACT_PAIRS:
        result = review_pair(source_for_sha[entry['parent_sha256']], source_for_sha[entry['candidate_sha256']], HAR)
        assert result['classification'] == 'preserving' and result['fingerprint_complete']
        assert result['residual_components'] == [] and result['changed_components'] == []
        assert result['settings']['quaternary_omission_coefficients'] == {'before': entry['before_quaternary_omissions'], 'after': entry['after_quaternary_omissions']}
        assert result['reviewer'] == 'har-b03-c2-quaternary-reset-omission-exact-pairs-v1'
    first = EXACT_PAIRS[0]
    changed = dict(source_for_sha[first['candidate_sha256']])
    changed['train.py'] = changed['train.py'].replace('torch.arange(self.lstmHidden - 9, dtype=torch.long)', 'torch.arange(self.lstmHidden - 99, dtype=torch.long)', 1)
    assert exact_pair_review(source_for_sha[first['parent_sha256']], changed) is None
    wrong_parent = dict(source_for_sha[first['parent_sha256']])
    wrong_parent['train.py'] = wrong_parent['train.py'].replace('torch.arange(self.lstmHidden - 6, dtype=torch.long)', 'torch.arange(self.lstmHidden - 98, dtype=torch.long)', 1)
    assert exact_pair_review(wrong_parent, source_for_sha[first['candidate_sha256']]) is None


def test_har_b03_c2_tertiary_reset_omission_ledger_is_hash_bound_and_complete():
    from pathlib import Path
    from experiments.ontology_har_tertiary_reset_omission import EXACT_PAIRS, review_pair as exact_pair_review
    from experiments.ontology_review_recurrent import review_pair, source_sha
    root = _fixture_repository_root() / ('data/c0c3/uci-har-pareto-v21/runs/' 'uci-har-pareto-v21-uci-har-pareto-openevolve-b03-c2/candidates')
    source_for_sha = {}
    for directory in root.iterdir():
        if directory.is_dir():
            source = {file.relative_to(directory).as_posix(): file.read_text(encoding='utf-8-sig') for file in directory.rglob('*.py')}
            source_for_sha[source_sha(source)] = source
    assert [entry['proposal'] for entry in EXACT_PAIRS] == [120, 122, 123, 124, 125, 126]
    for entry in EXACT_PAIRS:
        result = review_pair(source_for_sha[entry['parent_sha256']], source_for_sha[entry['candidate_sha256']], HAR)
        assert result['classification'] == 'preserving' and result['fingerprint_complete']
        assert result['residual_components'] == [] and result['changed_components'] == []
        assert result['settings']['tertiary_omission_coefficients'] == {'before': entry['before_tertiary_omissions'], 'after': entry['after_tertiary_omissions']}
        assert result['reviewer'] == 'har-b03-c2-tertiary-reset-omission-exact-pairs-v1'
    first = EXACT_PAIRS[0]
    changed = dict(source_for_sha[first['candidate_sha256']])
    changed['train.py'] = changed['train.py'].replace('torch.arange(self.lstmHidden - 9, dtype=torch.long)', 'torch.arange(self.lstmHidden - 99, dtype=torch.long)', 1)
    assert exact_pair_review(source_for_sha[first['parent_sha256']], changed) is None
    wrong_parent = dict(source_for_sha[first['parent_sha256']])
    wrong_parent['train.py'] = wrong_parent['train.py'].replace('torch.arange(self.lstmHidden - 6, dtype=torch.long)', 'torch.arange(self.lstmHidden - 98, dtype=torch.long)', 1)
    assert exact_pair_review(wrong_parent, source_for_sha[first['candidate_sha256']]) is None


def test_har_b03_c2_secondary_reset_omission_ledger_is_hash_bound_and_complete():
    from pathlib import Path
    from experiments.ontology_har_secondary_reset_omission import EXACT_PAIRS, review_pair as exact_pair_review
    from experiments.ontology_review_recurrent import review_pair, source_sha
    root = _fixture_repository_root() / ('data/c0c3/uci-har-pareto-v21/runs/' 'uci-har-pareto-v21-uci-har-pareto-openevolve-b03-c2/candidates')
    source_for_sha = {}
    for directory in root.iterdir():
        if directory.is_dir():
            source = {file.relative_to(directory).as_posix(): file.read_text(encoding='utf-8-sig') for file in directory.rglob('*.py')}
            source_for_sha[source_sha(source)] = source
    assert [entry['proposal'] for entry in EXACT_PAIRS] == [110, 111, 112, 113, 114, 115]
    for entry in EXACT_PAIRS:
        result = review_pair(source_for_sha[entry['parent_sha256']], source_for_sha[entry['candidate_sha256']], HAR)
        assert result['classification'] == 'preserving' and result['fingerprint_complete']
        assert result['residual_components'] == [] and result['changed_components'] == []
        assert result['settings']['secondary_omission_coefficients'] == {'before': entry['before_secondary_omissions'], 'after': entry['after_secondary_omissions']}
        assert result['reviewer'] == 'har-b03-c2-secondary-reset-omission-exact-pairs-v1'
    first = EXACT_PAIRS[0]
    changed = dict(source_for_sha[first['candidate_sha256']])
    changed['train.py'] = changed['train.py'].replace('torch.arange(self.lstmHidden - 10, dtype=torch.long)', 'torch.arange(self.lstmHidden - 99, dtype=torch.long)', 1)
    assert exact_pair_review(source_for_sha[first['parent_sha256']], changed) is None
    wrong_parent = dict(source_for_sha[first['parent_sha256']])
    wrong_parent['train.py'] = wrong_parent['train.py'].replace('torch.arange(self.lstmHidden - 6, dtype=torch.long)', 'torch.arange(self.lstmHidden - 98, dtype=torch.long)', 1)
    assert exact_pair_review(wrong_parent, source_for_sha[first['candidate_sha256']]) is None
