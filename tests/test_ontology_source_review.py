from experiments.review_ontology_sources import analyze, read_sources
from experiments.adjudicate_ontology_reviews import graph_job


def test_rooted_review_ignores_unused_alternative_architecture():
    source = '''
from torch import nn
class Unused(nn.Module):
    def __init__(self):
        self.rnn = nn.LSTM(8, 8)
class Active(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(1, 8, 3)
    def forward(self, x):
        return self.conv(x)
def build_model():
    return Active()
'''
    profile = analyze({'train.py': source}, 'fashion')
    assert profile['roots'] == ['Active']
    assert profile['fingerprint']['mixing'] == 'Conv2d'
    assert profile['fingerprint']['state'] is None


def test_agent_comment_cannot_create_rotary_or_gating_evidence():
    source = '''
from torch import nn
class Model(nn.Module):
    """RoPE SwiGLU new ontology: rotate_half(q)."""
    def __init__(self):
        self.linear = nn.Linear(8, 8)
    def forward(self, x):
        # apply_rotary_emb(x)
        return self.linear(x)
def build_model():
    return Model()
'''
    profile = analyze({'train.py': source}, 'addition')
    assert profile['fingerprint']['position'] is None
    assert profile['fingerprint']['feedforward'] is None
    assert profile['fingerprint']['mixing'] == 'dense affine network'


def test_custom_single_gate_has_equation_evidence():
    source = '''
from torch import nn
class Model(nn.Module):
    def __init__(self):
        self.transition = nn.Linear(12, 16)
    def recurrent_step(self, frame, state):
        hidden = state
        retention = torch.sigmoid(self.transition(frame))
        proposal = torch.tanh(frame)
        hidden = retention * hidden + (1.0 - retention) * proposal
        return hidden
def build_model():
    return Model()
'''
    profile = analyze({'train.py': source}, 'kws')
    assert profile['fingerprint']['state_update'] == 'single-gate interpolation recurrence'
    assert profile['evidence']['state_update'][0]['line'] == 10


def test_restored_src_layout_and_missing_artifact(tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    (source / 'model.py').write_text('x=1')
    assert read_sources(tmp_path) == {'src/model.py': 'x=1'}
    missing = analyze({}, 'addition')
    assert missing['family'] is None
    assert all(v is None for v in missing['fingerprint'].values())


def test_shape_review_preserves_factory_and_encoding_changes():
    source = '''
from torch import nn
MODE = "last"
class Model(nn.Module):
    def __init__(self, mode="last"):
        self.mode = mode
        self.linear = nn.Linear(8, 8)
    def forward(self, x):
        return self.linear(x)
def build_model():
    return Model(mode=MODE)
def encode_inputs(x):
    return x
'''
    shape = graph_job(('a',{'train.py':source}))[1]
    assert shape != graph_job(('b',{'train.py':source.replace('MODE = "last"','MODE = "mean"')}))[1]
    assert shape != graph_job(('b',{'train.py':source.replace('return x','return x.flip(-1)')}))[1]


def test_scalar_pool_mixture_is_not_input_dependent_channel_gating():
    source = '''
from torch import nn
class MixedPool(nn.Module):
    def __init__(self):
        self.max_logit = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        max_weight = torch.sigmoid(self.max_logit)
        return max_weight * F.max_pool2d(x, 2) + (1.0-max_weight) * F.avg_pool2d(x, 2)
class ImageClassifier(nn.Module):
    def __init__(self):
        self.pool = MixedPool()
    def forward(self, x):
        return self.pool(x)
def build_model():
    return ImageClassifier()
'''
    assert analyze({'train.py':source},'fashion')['fingerprint']['channel_interaction'] is None
