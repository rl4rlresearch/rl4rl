"""Finite SHA-bound reviews for the source-readable Addition b05-c3 relative-score cohort."""
from copy import deepcopy
from experiments.ontology_addition_b03_references import GAUGE
from experiments.ontology_addition_b03c1_references import result as changing_result

RELATIVE = deepcopy(GAUGE)
RELATIVE.update(
    attention='position-only learned softmax scores', position='none', relative_scores=True,
    relative_score_layout=None, score_bias='relative-distance score table',
)


def references():
    pairs = [
        ('9046985cade0a5e8c1cb2c9015cf1462a13bca106f8c4fcd5ca33578fbce53ad', '6de275ca4fc05ec6784266da8d7d065cb829a5f4128dafa1304963013154f02b', ['position', 'parameter_construction'], 'The per-head cyclic relative displacement changes from learned parameters to a fixed source buffer.', 'self.register_buffer("relative_shift", torch.arange(1, n_head, dtype=torch.float32) * (max_seq_len / n_head))', {'position': 'no additive position path; fixed cyclic per-head displacement into learned relative-distance score table', 'attention_scores': 'position-only softmax scores with fixed cyclic per-head displacement and lower-triangular causal routing', 'parameter_construction': 'learned relative-distance kernel with fixed head displacement coordinates'}),
        ('66684f4287836a7ebaee3fb57361138dcacf5a5dea6d94aed6adf9d34527a2c3', 'c85eabeedef6aaa647f77fa92eb0272600c90c737edc1ea9bc316a5d6b5a82a8', ['position', 'attention_scores', 'parameter_construction'], 'The relative kernel uses fixed head spacing and a common temperature; its two terminal logits are tied instead of independent.', 'self.relative_bias = nn.Parameter(torch.zeros(max_seq_len - 2))', {'position': 'no additive position path; fixed head spacing with a tied-terminal relative-distance score table', 'attention_scores': 'position-only common-temperature softmax scores with fixed cyclic displacement and tied terminal kernel logits', 'parameter_construction': 'learned shared relative kernel with two terminal coordinates tied and fixed head displacement'}),
        ('a159cdc371cebf0125bb200bb54663edddc5a89c7494e98c438e5220afe3bfea', '5f3f93092b44b70f9e4532b0455bb679bcafa9b6baec979c48c7061275d54fec', ['position', 'attention_scores', 'parameter_construction', 'normalization'], 'All MLP-scale LayerNorm coordinates become fixed while the relative kernel adopts fixed head spacing, common temperature, and three tied terminal logits.', 'self.register_buffer("fixed_weight", torch.ones(d_model))', {'position': 'no additive position path; fixed head spacing with a three-way-tied terminal relative-distance score table', 'attention_scores': 'position-only common-temperature softmax scores with fixed cyclic displacement and three tied terminal kernel logits', 'parameter_construction': 'learned shared relative kernel with three terminal coordinates tied and fixed head displacement', 'normalization': 'fixed-scale GaugeFixedMLPScaleLayerNorm and LayerNorm'}),
        ('f692a0b2e61b4607a71188602df1e73e5d859e1ffe37c374a2bb26fab5e236a1', '0c15d067604d915b34ad0eb97e024b9fc3cde3ddbd642867026ce532a4764eee', ['embedding', 'sharing', 'normalization', 'projection_relations'], 'Embedding rows are stored modulo per-token hidden shifts, final normalization gains a tied zero-sum bias, and values become one head-shared encoder repeated over routes.', 'return shared_value.repeat(1, 1, self.n_head)', {'embedding': 'token lookup table stored modulo one hidden-coordinate shift per token', 'sharing': 'one learned value encoder repeated over explicit attention heads', 'normalization': 'TiedFinalScaleLayerNorm with shared scale and two-way-tied zero-sum bias', 'projection_relations': 'head-shared value map followed by the existing projection path'}),
    ]
    return {('openevolve_v21', before, after): changing_result(RELATIVE, changed, note, code, updates) for before, after, changed, note, code, updates in pairs}
