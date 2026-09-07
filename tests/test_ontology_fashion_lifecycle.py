"""Exact reviewed lifecycle programs must not generalize to new computation."""
import json
from pathlib import Path

from experiments.ontology_review_vision_lm import profile, review_pair


FASHION = 'openevolve_v21_fashion_mnist'
CATALOG = Path(__file__).parents[1] / 'experiments/ontology_fashion_lifecycle_references.json'


def _references():
    return json.loads(CATALOG.read_text(encoding='utf8'))['reviewed_profiles']


def _source(group):
    ref = next(r for r in _references() if r['reference_group'] == group)
    return {'train.py': '\n\n'.join(ref['program_parts'].values())}


def test_all_complete_lifecycle_programs_replay_with_source_evidence():
    for ref in _references():
        source = _source(ref['reference_group'])
        result = review_pair(source, source, FASHION)
        assert result['classification'] == 'preserving', ref['reference_group']
        assert result['fingerprint_complete']
        assert len(result['fingerprint']) == 27
        assert all(isinstance(value, str) and value for value in result['fingerprint'].values())
        assert result['evidence']
        assert result['training']['lifecycle']
        assert result['inference']['procedure']


def test_ema_lifecycle_variations_do_not_invent_model_memory():
    result = review_pair(_source(266), _source(335), FASHION)
    assert result['classification'] == 'preserving'
    assert 'floating buffers' in result['training']['lifecycle']
    assert 'no recurrent feature state' in result['fingerprint']['state']
    disabled_dropout = profile(_source(374), FASHION)
    assert disabled_dropout['fingerprint']['stochasticity'] == 'none: declared Dropout has p=0'


def test_regional_readout_statistics_remain_distinct_from_standard_head():
    result = review_pair(_source(374), _source(421), FASHION)
    assert result['classification'] == 'changing'
    assert result['family_signature']['summary_basis'] == ['maximum', 'mean']
    assert '3x3' in result['fingerprint']['spatial_readout']
    mixed = profile(_source(100), FASHION)
    assert '2x2' in mixed['fingerprint']['spatial_readout']
    assert '1x1 global-maximum' in mixed['fingerprint']['spatial_readout']


def test_batchnorm_fold_is_preserving_but_changed_algebra_is_unreviewed():
    result = review_pair(_source(255), _source(514), FASHION)
    assert result['classification'] == 'preserving'
    assert 'W_fused=scale*W' in result['fingerprint']['parameter_construction']
    assert 'cache' in result['fingerprint']['state']
    original = _source(514)
    changed = {name: text.replace('torch.rsqrt(', 'torch.sqrt(') for name, text in original.items()}
    assert changed != original
    assert profile(changed, FASHION) is None


def test_reviewed_lifecycle_does_not_admit_signal_mutation_or_extra_state():
    original = _source(255)
    changed = {name: text.replace('self.classifier(self.features(images))', 'self.classifier(self.features(images * images))') for name, text in original.items()}
    assert changed != original
    assert profile(changed, FASHION) is None
    extra_state = {name: text.replace('was_training = self.training', 'self.feature_memory = []\n        was_training = self.training') for name, text in original.items()}
    assert extra_state != original
    assert profile(extra_state, FASHION) is None
