from experiments.ontology_semantics import fingerprint_program, constant
from experiments.ontology_transition_review import settings_signature
from experiments.ontology_affine_readout import affine_readout_signature
import ast


BASE='''
from torch import nn
import torch.nn.functional as F
WIDTH=24
class Model(nn.Module):
    def __init__(self, aggregation="last"):
        self.aggregation=aggregation
        self.linear=nn.Linear(20, WIDTH)
    def forward(self,x):
        if self.aggregation=="mean":
            x=x.mean(dim=1)
        else:
            x=x[:,-1,:]
        return self.linear(x)
def build_model():
    return Model(aggregation="mean")
'''


def fp(source):
    return fingerprint_program({'train.py':source})


def test_factory_branch_selected_and_dimensional_change_preserving():
    a=fp(BASE)
    assert a['config_decisions'][0]['value'] is True
    assert a['shape']==fp(BASE.replace('WIDTH=24','WIDTH=32'))['shape']
    assert a['shape']!=fp(BASE.replace('return Model(aggregation="mean")','return Model(aggregation="last")'))['shape']


def test_scalar_runtime_change_is_not_erased_as_width():
    source=BASE.replace('self.aggregation=aggregation','self.scale=2\n        self.aggregation=aggregation').replace('return self.linear(x)','return self.linear(x)*self.scale')
    assert fp(source)['shape']!=fp(source.replace('self.scale=2','self.scale=3'))['shape']


def test_constant_interpreter_does_not_execute_source():
    assert not isinstance(constant(ast.parse('__import__("os").system("echo bad")',mode='eval').body,{}),int)


def test_input_algebra_not_confused_with_parameter_coordinates():
    a=BASE.replace('return self.linear(x)','return self.linear(x*x)')
    assert fp(BASE)['shape']!=fp(a)['shape']


def test_bias_flag_is_not_a_primitive_change():
    assert fp(BASE)['shape']==fp(BASE.replace('nn.Linear(20, WIDTH)','nn.Linear(20, WIDTH, bias=False)'))['shape']


def test_custom_recurrent_equations_survive_normalization():
    source=BASE.replace('return self.linear(x)','return gate*x + (1-gate)*hidden')
    assert fp(source)['shape']!=fp(source.replace('gate*x + (1-gate)*hidden','gate*hidden + (1-gate)*x'))['shape']


def test_input_dependent_lookup_is_not_erased_as_parameter_only():
    source=BASE.replace('self.aggregation=aggregation','self.table=nn.Parameter(torch.zeros(10,20))\n        self.aggregation=aggregation').replace('return self.linear(x)','z=self.table[x]\n        return self.linear(z)')
    assert fp(source)['shape']!=fp(source.replace('z=self.table[x]','z=self.table[x+1]'))['shape']


def test_unrecognized_buffer_construction_remains_visible():
    source=BASE.replace('self.aggregation=aggregation','self.register_buffer("basis", torch.sin(torch.arange(20)))\n        self.aggregation=aggregation').replace('return self.linear(x)','return self.linear(x @ self.basis)')
    assert fp(source)['shape']!=fp(source.replace('torch.sin','torch.cos'))['shape']


def test_algebraically_generated_weight_is_not_plain_free_affine():
    source=BASE.replace('self.aggregation=aggregation','self.factor=nn.Parameter(torch.ones(24,20))\n        self.aggregation=aggregation').replace('return self.linear(x)','return F.linear(x,self.linear.weight*self.factor)')
    assert fp(source)['shape']!=fp(source.replace('self.linear.weight*self.factor','self.linear.weight'))['shape']


def test_fixed_schedule_variants_are_settings_but_signal_routing_is_not():
    source=BASE.replace('    def forward(self,x):','    def frame_schedule(self,available_frames):\n        return list(range(available_frames))\n    def forward(self,x):')
    shorter=source.replace('return list(range(available_frames))','return [i*2 for i in range(min(20,available_frames//2))]')
    assert fp(source)['shape']==fp(shorter)['shape']
    adaptive=source.replace('return list(range(available_frames))','return self.policy(self.signal)')
    assert fp(source)['shape']!=fp(adaptive)['shape']


def test_numeric_threshold_tuning_keeps_operator_and_index_structure():
    def sig(s):return settings_signature({'train.py':s})
    source=BASE.replace('return self.linear(x)','return self.linear(x) > 0.7')
    assert sig(source)==sig(source.replace('> 0.7','> 0.8'))
    assert sig(source)!=sig(source.replace('> 0.7','< 0.7'))
    assert sig(source)!=sig(source.replace('x.mean(dim=1)','x.mean(dim=2)'))
    assert sig(source)!=sig(source.replace('> 0.7','> 0.0'))


def test_powers_positional_axes_and_decorators_are_not_numeric_settings():
    def sig(s):return settings_signature({'train.py':s})
    for before,after in [('x**1','x**2'),('x.pow(1)','x.pow(2)'),('x.mean(1)','x.mean(2)'),('x.chunk(2)','x.chunk(3)')]:
        source=BASE.replace('return self.linear(x)','return '+before)
        assert sig(source)!=sig(source.replace(before,after))
    decorated=BASE.replace('    def forward(self,x):','    @custom_transform\n    def forward(self,x):')
    assert sig(BASE)!=sig(decorated)
    assert fp(BASE)['shape']!=fp(decorated)['shape']


def test_main_function_config_call_cannot_change_silently():
    source=BASE.replace('def build_model():\n    return Model(aggregation="mean")','def main():\n    return Model(aggregation="mean")')
    other=source.replace('return Model(aggregation="mean")','return Model(aggregation="last")')
    assert fp(source)['shape']!=fp(other)['shape']
    assert settings_signature({'train.py':source})!=settings_signature({'train.py':other})


def test_method_defaults_and_argument_shadowing_are_preserved():
    source=BASE.replace('def forward(self,x):','def forward(self,x,aggregation="mean"):').replace('if self.aggregation=="mean":','if aggregation=="mean":')
    assert not fp(source)['config_decisions']
    other=source.replace('def forward(self,x,aggregation="mean")','def forward(self,x,aggregation="last")')
    assert fp(source)['shape']!=fp(other)['shape']
    assert settings_signature({'train.py':source})!=settings_signature({'train.py':other})


def test_affine_readout_rewrite_preserves_features_but_not_nonlinear_readout():
    source='''
from torch import nn
class Model(nn.Module):
    def __init__(self):
        self.classifier=nn.Linear(8,8)
    def classify(self,state):
        hidden,summary=state
        features=torch.cat((hidden,summary),dim=-1)
        return self.classifier(features)
def build_model():
    return Model()
'''
    other=source.replace('self.classifier=nn.Linear(8,8)','self.first_classifier=nn.Linear(8,4)\n        self.last_classifier=nn.Linear(7,4)').replace('return self.classifier(features)','return torch.cat((self.first_classifier(features),self.last_classifier(features[:,:-1])),dim=-1)')
    def sig(s):return affine_readout_signature({'train.py':s})
    assert sig(source) and sig(source)==sig(other)
    assert sig(source)!=sig(other.replace('self.first_classifier(features)','torch.relu(self.first_classifier(features))'))
    assert sig(source)!=sig(source.replace('hidden,summary),dim','hidden,summary.mean(dim=1)),dim'))
