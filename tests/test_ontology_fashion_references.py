import json
from pathlib import Path

from experiments.ontology_review_vision_lm import FASHION, profile, review_pair
from experiments.review_ontology_sources import CORE, TASK_KEYS


def references():
    path = Path(__file__).parents[1] / 'experiments' / 'ontology_fashion_custom_references.json'
    return json.loads(path.read_text(encoding='utf-8'))['reviewed_profiles']


def source(ref):
    return {'train.py': '\n\n'.join(ref['program_parts'].values()) +
            '\n\ndef build_model():\n    return ImageClassifier()\n'}


def test_direct_model_profiles_replay_with_every_component():
    required = CORE + TASK_KEYS['fashion'].split()
    for ref in references():
        result = profile(source(ref), FASHION)
        assert result, ref['reference_proposal']
        assert result['fingerprint'] == ref['fingerprint']
        assert all(result['fingerprint'].get(k) for k in required)


def test_gate_statistics_change_but_projection_untying_preserves_family():
    refs = references()
    base = source(refs[0])
    std_added = source(refs[38])
    untied = source(refs[39])
    assert review_pair(base, std_added, FASHION)['classification'] == 'changing'
    result = review_pair(base, untied, FASHION)
    assert result['classification'] == 'preserving'
    assert result['fingerprint']['sharing'] != profile(base, FASHION)['fingerprint']['sharing']


def test_fixed_view_calibration_preserves_but_confidence_routing_changes():
    refs = references()
    assert review_pair(source(refs[2]), source(refs[3]), FASHION)['classification'] == 'preserving'
    assert review_pair(source(refs[0]), source(refs[4]), FASHION)['classification'] == 'changing'


def test_fixed_pointwise_probability_fusion_is_an_inference_setting():
    refs = references()
    result = review_pair(source(refs[170]), source(refs[187]), FASHION)
    assert result['classification'] == 'preserving'
    assert result['inference'] != profile(source(refs[170]), FASHION)['inference']


def test_unreviewed_signal_operation_cannot_reuse_reference():
    base = source(references()[0])
    changed = {k: v.replace('torch.sigmoid(channel_gate)', 'torch.sin(channel_gate)') for k, v in base.items()}
    assert profile(changed, FASHION) is None
