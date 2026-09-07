"""Finite SHA-bound reviews for the source-readable Addition b03-c2 gauge cohort."""
from copy import deepcopy
from experiments.review_ontology_sources import CORE, TASK_KEYS

GAUGE = {
    'operand_representation': 'ordered base-10 operand-pair token sequence; unchanged incoming digit order',
    'target_representation': ['decimal-digit token targets', 'direct decimal-digit output'],
    'embedding': 'token lookup table',
    'position': 'additive',
    'relative_scores': False,
    'attention': 'content-dependent softmax QK scores',
    'feedforward': 'ungated pointwise MLP',
    'normalization_modules': ['LayerNorm'],
    'normalization_topology': [],
    'relative_score_layout': None,
    'parameter_generator': [],
    'attention_kernel': [],
    'score_bias': 'none',
}


def result(note, code):
    fingerprint = dict.fromkeys(CORE + TASK_KEYS['addition'].split(), 'source-traced causal transformer mechanism')
    fingerprint.update(
        input_units='operand-pair and generated decimal-digit tokens',
        input_transform='ordered base-10 digit-column packing; token alphabet/delimiters are coordinates',
        embedding='token lookup table', position='additive position features',
        mixing='causal transformer attention',
        routing='content-dependent softmax QK scores; lower-triangular causal routing',
        state='feedforward hidden activations; no persistent recurrent state in traced methods',
        feedforward='ungated pointwise MLP',
        parameter_construction='learned affine coefficients and lookup coordinates',
        sharing='self.lm_head.weight = self.token_emb.weight', normalization='LayerNorm',
        connectivity='residual attention and pointwise transformation paths',
        aggregation='per-position attention-mixed features for token prediction',
        output='affine vocabulary readout', symmetry='ordered operand representation; no explicit min/max quotient in the embedding path',
        conditional_compute='fixed forward computation; no persistent-state writes or untraced dynamic branch',
        activation='gelu', stochasticity='dropout modules (rate is a setting)',
        bottleneck='no separate sequential factor bottleneck',
        iteration='fixed forward block stack; autoregressive token iteration is inference',
        other='all runtime operations are accounted for by the source-traced signal/coordinate graph',
        operand_encoding='ordered base-10 digit-column pair tokens', digit_order='least-significant column first; unchanged input order',
        carry_representation='implicit in attention/hidden features; no explicit persistent carry register',
        attention_scores='content-dependent softmax QK scores; lower-triangular causal routing',
        projection_relations='learned affine value/output maps and Q/K maps; stored-coordinate constraints recorded separately',
        output_factorization='autoregressive decimal-digit token logits; optional constant termination token',
    )
    return {
        'classification': 'preserving', 'fingerprint': fingerprint, 'fingerprint_complete': True,
        'family_signature': deepcopy(GAUGE), 'changed_components': [], 'notes': note,
        'evidence': [{'file': 'src/model.py', 'scope': 'complete before/after source review',
                      'kind': 'directly source-adjudicated b03-c2 coordinate-gauge transition', 'code': code}],
        'reviewer': 'addition-source-pair-adjudication-v11',
    }


def references():
    pairs = [
        ('fb3e23e32f702859e0a86c61242c4d2205f17e2ebe15e8fffdeed6d607b25202', '76f75cfaf53296af7704b78018eb525eab5e4b06f9242428c0abe59d6c13f7a5', 'Value/output chart constraints and final affine-norm coordinates are reconstructed in the same causal attention path.', 'self.proj_head_weight = nn.Parameter(torch.empty(d_model - 4))'),
        ('fc281c7499b4ef19cf8cb3999e444ad657bfe5f5babc3e4e54c413f2efdc5955', '6cb67335ab4dab7120d4cb42caef0214d16b492e2aa2588f7a4ebe04d654c312', 'Additional pivoted projection charts parameterize the existing value/output basis without adding a signal route.', 'self.proj_first_weight = nn.Parameter(torch.empty(d_model - 2))'),
        ('ae366bb775f74bc6097aee802d5091f620c08801cd672f51cee8096ebe407043', 'c79faece6587a142ccfdc99e33f4e535199ee6915bbfb0015ffdd55172c47382', 'Independent head-row scale gauges are reconstructed before the unchanged attention projection.', 'head_pivot = int(self.proj_head_pivot.item())'),
        ('1ce08edf84240976e0bbbbcf3842818e758e71650dce2a277371707d881d26f1', '9300203be317a8cdcff7a3cfcf6fbd75f4350d548d159e5eae127f214c7b2909', 'Pivoted query-coordinate charts preserve the existing QK attention computation and causal routing.', 'q_target_pivot = int(self.q_target_pivot.item())'),
        ('fc281c7499b4ef19cf8cb3999e444ad657bfe5f5babc3e4e54c413f2efdc5955', '9df08731d47fa12a8337cce23a1d786d9c9a9c938f9bf53eeb6793005ad2378a', 'Head and terminal query coordinates are stored in a gauge chart while the source retains the same QK mechanism.', 'self.q_head_weight = nn.Parameter(torch.empty(d_model - 3))'),
        ('fc281c7499b4ef19cf8cb3999e444ad657bfe5f5babc3e4e54c413f2efdc5955', 'd206be16e82824217b161852001922ffeac7dd4932f0b6a52bf44e3a206f0e7d', 'Penultimate and terminal query gauges preserve the learned affine Q/K maps and causal softmax path.', 'self.q_penultimate_weight = nn.Parameter(torch.empty(d_model - 3))'),
        ('bcd7c7e666d8b389b30fed069432dc0f86044ee02bf75a958980a0c0041d8a0c', '8eb969d4e360364c14824e0485fbfa6f495447c0a64ca1b0a7cc74834c2f3d75', 'A neighboring output-row orthogonal chart is reconstructed into the existing attention projection.', 'self.proj_neighbor_weight = nn.Parameter(torch.empty(d_model - 2))'),
        ('bcd7c7e666d8b389b30fed069432dc0f86044ee02bf75a958980a0c0041d8a0c', '31e85c866ac6bba3da19a96ac7da6de0b58a616710f8e7265df76bdb6896eaff', 'A biased query-row orthogonal chart is folded into the same Q/K affine coordinates.', 'self.q_biased_weight = nn.Parameter(torch.empty(d_model - 2))'),
        ('b06c16797e3c7e909b6ebb07b75e147faea0d22b4d0303a50081c291db075e0c', 'dcea0feb6f06494c772939d4ba00cec24c48c53e545089f80ad5bfba6ec9832d', 'Neighbor and biased query charts add stored coordinates only; the reconstructed causal QK path is unchanged.', 'q_neighbor_pivot = int(self.q_neighbor_pivot.item())'),
        ('b06c16797e3c7e909b6ebb07b75e147faea0d22b4d0303a50081c291db075e0c', 'd5d663b1d47466ad07685e667947b8c2fd9838fb7ab120fb14e198cbe62d6c79', 'A normalized query frame and feature-centered final bias chart retain the same causal transformer mechanism.', 'self.ln_f_bias = nn.Parameter(torch.zeros(cfg.d_model - 1))'),
    ]
    return {('openevolve_v21', before, after): result(note, code) for before, after, note, code in pairs}
