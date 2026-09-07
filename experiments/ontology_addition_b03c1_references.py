"""Finite SHA-bound reviews for the source-readable Addition b03-c1 mechanism cohort."""
from copy import deepcopy
from experiments.ontology_addition_b03_references import GAUGE, result as preserving_result

BILINEAR = deepcopy(GAUGE)
BILINEAR.update(
    attention='content-dependent softmax affine-bilinear scores',
    attention_kernel=['direct learned affine bilinear score operator'],
)


def result(signature, changed, note, code, updates):
    reviewed = preserving_result(note, code)
    reviewed.update(
        classification='changing', family_signature=deepcopy(signature),
        changed_components=changed, reviewer='addition-source-pair-adjudication-v12',
    )
    reviewed['fingerprint'].update(updates)
    reviewed['evidence'][0]['kind'] = 'directly source-adjudicated b03-c1 mechanism transition'
    return reviewed


def references():
    pairs = [
        ('39087942babe73d8ff93eeecab5af3bfdb0b5e5743c88fc60d1c03fd3445feba', '330e17f478f159e5f7c2c37aab67a722676f1e6393767c6cc6c97c4f1fe5088d', GAUGE, ['sharing', 'projection_relations'], 'The per-head value maps are replaced by one shared zero-mean value readout replicated over independently routed heads.', 'v = v.unsqueeze(1).expand(-1, self.n_head, -1, -1)', {'sharing': 'head-shared zero-mean value readout replicated over independently routed heads', 'projection_relations': 'independent learned Q/K maps with one shared learned value map'}),
        ('e49ea9cd3ab25ddb944ed65d76e955066c5eb92e60db0bbd43e5d34394d5645c', '81d1374a61dc3c5f532716f9a44487590bb8220949787d2f95d3c5404e6d9c4c', GAUGE, ['sharing', 'projection_relations'], 'The key dictionary becomes head-shared while route specialization remains in independent query projections and relative biases.', 'k = k.unsqueeze(1).expand(-1, self.n_head, -1, -1)', {'sharing': 'independent query projections with head-shared zero-mean key and value readouts', 'projection_relations': 'independent learned query maps and head-shared learned key/value maps'}),
        ('1e5230e66d653b2d770180024d441e703e22f8f9e9bd92e53eba4532c7840456', '8d7bf395971451cbae1cb04f333184eaecb1a11d83c48e8f0105040bc67ee84f', BILINEAR, ['parameter_construction', 'attention_scores', 'projection_relations'], 'The factorized Q/K coordinates are replaced by a learned affine bilinear score operator per head; causal softmax routing remains but its score construction changes.', 'routed_query = torch.einsum("bti,hij->bhtj", affine_content, score_coeff)', {'attention_scores': 'direct learned affine-bilinear content scores; lower-triangular causal routing', 'parameter_construction': 'learned per-head affine bilinear score operators', 'projection_relations': 'unfactorized per-head score operators with a shared learned value map'}),
        ('f49f11dbca9a82bff4b6745cc102a7c81104db7e1db338e867b89ee9951fedd3', '94f827c917a9e04565f6d6bb8ac771d20f7a01c4904d64a2493793f0304c0c3e', GAUGE, ['sharing', 'aggregation', 'projection_relations'], 'Head-specific output maps become one shared value representation and contexts are commutatively summed before the learned projection.', 'y = y.sum(dim=1) / math.sqrt(self.n_head)', {'sharing': 'head-shared learned value representation with independent attention routes', 'aggregation': 'commutative variance-scaled sum across attention heads before output projection', 'projection_relations': 'shared learned value map followed by one learned output projection'}),
        ('f49f11dbca9a82bff4b6745cc102a7c81104db7e1db338e867b89ee9951fedd3', '08e38da63601b0ef164c4b09ade247ef6d613e05aabc4e294d78395652459946', GAUGE, ['aggregation', 'output', 'projection_relations'], 'The dense attention output projection is removed and replaced by learned per-head gains in distinct residual-channel slots.', 'y = y * self.proj_gain.view(1, self.n_head, 1, 1)', {'aggregation': 'gain-scaled routed contexts retained in distinct residual-channel slots', 'output': 'per-head gain-scaled routed contexts without a dense attention output projection', 'projection_relations': 'learned per-head scalar gains replace the dense attention output map'}),
    ]
    return {('openevolve_v21', before, after): result(signature, changed, note, code, updates) for before, after, signature, changed, note, code, updates in pairs}
