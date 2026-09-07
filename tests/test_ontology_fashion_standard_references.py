"""Finite source review covers ordinary computation without weakening guards."""
import json
import ast
from pathlib import Path

from experiments.ontology_review_vision_lm import FASHION, _fashion_gate_family, profile, review_pair


def _refs():
    path = Path(__file__).parents[1] / 'experiments/ontology_fashion_standard_references.json'
    return json.loads(path.read_text(encoding='utf8'))['reviewed_profiles']


def _source(group):
    ref = next(r for r in _refs() if r['reference_group'] == group)
    return {'train.py': '\n\n'.join(ref['program_parts'].values())}


def test_all_reviewed_standard_programs_have_complete_source_bound_profiles():
    for ref in _refs():
        source = _source(ref['reference_group'])
        result = review_pair(source, source, FASHION)
        assert result['classification'] == 'preserving'
        assert result['fingerprint_complete'] and len(result['fingerprint']) == 27
        assert result['evidence'] and result['inference']['procedure']
        assert all(isinstance(value, str) and value for value in result['fingerprint'].values())


def test_chunk_slice_and_separate_fixed_view_means_preserve_the_same_family():
    for group in [277, 430]:
        assert review_pair(_source(248), _source(group), FASHION)['classification'] == 'preserving'
    source = _source(248)
    routed = {name: text.replace('0.5 * (logits + flipped_logits)', 'logits.softmax(dim=1) * flipped_logits') for name, text in source.items()}
    assert routed != source and profile(routed, FASHION) is None


def test_group_sharing_and_ema_remain_detailed_without_a_new_family():
    grouped = profile(_source(220), FASHION)
    assert '12 groups at48 channels' in grouped['fingerprint']['channel_interaction']
    assert grouped['family_signature']['spatial_operator'] == 'learned local Conv2d'
    ema = profile(_source(299), FASHION)
    assert 'no recurrent feature state' in ema['fingerprint']['state']
    assert 'EMA' in ema['fingerprint']['state']
    changed = {name: text.replace('channels // 4', 'channels // 8') for name, text in _source(220).items()}
    assert changed != _source(220) and profile(changed, FASHION) is None


def test_positive_tanh_and_sigmoid_gates_share_a_family_with_formula_details():
    result = review_pair(_source(76), _source(363), FASHION)
    assert result['classification'] == 'preserving'
    assert '1+tanh' in profile(_source(76), FASHION)['fingerprint']['activation']
    assert '2*sigmoid' in result['fingerprint']['activation']
    assert result['family_signature']['channel_gate']['basis'] == ['mean']
    signed = {name: text.replace('features * (1.0 + modulation)', 'features * modulation') for name, text in _source(76).items()}
    assert signed != _source(76) and profile(signed, FASHION) is None


def test_multiscale_readouts_and_local_mean_skip_are_explicit_boundaries():
    assert review_pair(_source(220), _source(154), FASHION)['classification'] == 'changing'
    early = profile(_source(218), FASHION)
    assert 'two earlier depths' in early['fingerprint']['scale_representation']
    pyramid = review_pair(_source(238), _source(256), FASHION)
    assert pyramid['classification'] == 'preserving'
    assert '1x1,2x2,4x4' in pyramid['fingerprint']['scale_representation']
    parts = review_pair(_source(256), _source(418), FASHION)
    assert parts['classification'] == 'changing'
    assert parts['family_signature']['spatial_pooling'] == 'learned spatial softmax maps'
    assert 'four72-channel part vectors' in parts['fingerprint']['spatial_readout']
    assert 'channel modulation' in parts['fingerprint']['routing']
    wrong_axis = {name: text.replace('F.softmax(attention, dim=-1)', 'F.softmax(attention, dim=1)') for name, text in _source(418).items()}
    assert wrong_axis != _source(418) and profile(wrong_axis, FASHION) is None


def test_supplemental_exact_core_composes_with_reviewed_inference_coefficients():
    original = _source(238)
    calibrated = {name: text.replace('/ 0.75317', '/ 0.81234') for name, text in original.items()}
    assert calibrated != original
    result = review_pair(original, calibrated, FASHION)
    assert result['classification'] == 'preserving'
    assert any(item.get('kind') == 'exact custom model with source-audited inference settings' for item in result['evidence'])
    changed_core = {name: text.replace('pyramid.append(F.adaptive_avg_pool2d(features, output_size)', 'pyramid.append(F.adaptive_max_pool2d(features, output_size)') for name, text in calibrated.items()}
    assert changed_core != calibrated and profile(changed_core, FASHION) is None
    invalid = {name: text.replace('/ 0.81234', '/ 0.0') for name, text in calibrated.items()}
    assert invalid != calibrated and profile(invalid, FASHION) is None


def test_high_group_profiles_are_complete_source_bound_and_nonoverlapping():
    """The high-group sweep is a finite exact catalog, never a template."""
    references = _refs()
    # The pre-existing supplemental catalog has 13 profiles.  Each source-read
    # residual program in this sweep carries its own complete-diff evidence.
    high_groups = [
        ref for ref in references
        if any(item.get('kind') == 'complete exact-program diff review' for item in ref['evidence'])
    ]
    assert len(high_groups) == 117
    assert len(references) == 130
    custom_path = Path(__file__).parents[1] / 'experiments/ontology_fashion_custom_references.json'
    custom_programs = {
        ref['program_sha256']
        for ref in json.loads(custom_path.read_text(encoding='utf8'))['reviewed_profiles']
    }
    for ref in high_groups:
        assert ref['program_sha256'] not in custom_programs
        assert set(ref['fingerprint']) and len(ref['fingerprint']) == 27
        assert any(
            item.get('kind') == 'complete exact-program diff review'
            and item.get('program_sha256') == ref['program_sha256']
            and item.get('source_sha256') == ref['source_sha256']
            for item in ref['evidence']
        )
        for part in ref['program_parts'].values():
            ast.parse(part)


def test_gate_and_spatial_context_admission_stays_narrow():
    """Known normalizations share a family, but nearby mechanisms do not."""
    tanh_gate = _source(542)
    sigmoid_gate = _source(480)
    tanh_profile = next(ref for ref in _refs() if ref['reference_group'] == 542)
    sigmoid_profile = next(ref for ref in _refs() if ref['reference_group'] == 480)
    assert tanh_profile['family_signature']['channel_gate']['basis'] == ['mean']
    assert tanh_profile['family_signature']['channel_gate'] != sigmoid_profile['family_signature']['channel_gate']
    assert tanh_profile['family_signature']['channel_gate']['operator'] == 'affine pointwise affine tanh gain'
    assert '1+0.5*tanh' in tanh_profile['fingerprint']['activation']
    assert '2*sigmoid' in sigmoid_profile['fingerprint']['activation']
    assert _fashion_gate_family({'operator': 'affine GELU affine tanh gain', 'basis': ['mean']}) == {
        'operator': 'affine pointwise affine tanh gain', 'basis': ['mean']
    }
    assert _fashion_gate_family({'operator': 'affine GELU affine tanh gain', 'basis': ['mean', 'maximum']}) == {
        'operator': 'affine pointwise affine tanh gain',
        'basis': ['mean', 'maximum'],
    }
    signed = {name: text.replace('1.0 + 0.5 * gate', '1.0 - 0.5 * gate') for name, text in tanh_gate.items()}
    assert signed != tanh_gate
    assert all('1.0 + 0.5 * gate' not in part for part in signed.values())

    mixed_pool = _source(565)
    changed_context = {
        name: text.replace('return mix * maximum + (1.0 - mix) * average', 'return mix * maximum + average')
        for name, text in mixed_pool.items()
    }
    assert changed_context != mixed_pool
    assert all('mix * maximum + (1.0 - mix) * average' not in part for part in changed_context.values())
