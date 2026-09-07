import json
from pathlib import Path

from experiments.ontology_fashion_composition import reviewed_composition, composition_key
from experiments.ontology_review_vision_lm import _inference_key, _inference_reference


def parts():
    path = Path(__file__).parents[1] / 'experiments' / 'ontology_fashion_custom_references.json'
    return json.loads(path.read_text())['reviewed_profiles'][170]['program_parts']


def review(source):
    return reviewed_composition(source, _inference_key, _inference_reference)


def test_custom_core_and_audited_fixed_view_coefficients_compose():
    source = parts()
    changed = {k: v.replace('(3.0, 2.0, 2.0, 2.0, 2.0)', '(3.25, 2.0, 2.0, 2.0, 2.0)') for k, v in source.items()}
    assert changed != source
    before, after = review(source), review(changed)
    assert before and after
    assert after['family_signature'] == before['family_signature']
    assert after['fingerprint'] == before['fingerprint']
    assert after['settings'] != before['settings']


def test_no_custom_model_or_inference_operator_is_erased_by_composition():
    source = parts()
    changes = [('detail_kernels.mean', 'detail_kernels.amax'),
               ('dim=0).sum(dim=0)', 'dim=0).amax(dim=0)'),
               ('(3.0, 2.0, 2.0, 2.0, 2.0)', '(0.0, 2.0, 2.0, 2.0, 2.0)')]
    for old, new in changes:
        changed = {k: v.replace(old, new) for k, v in source.items()}
        assert changed != source, old
        assert review(changed) is None, old


def test_inline_training_core_keeps_exact_computation_with_tunable_inference_weights():
    path = Path(__file__).parents[1] / 'experiments' / 'ontology_fashion_custom_references.json'
    source = json.loads(path.read_text())['reviewed_profiles'][300]['program_parts']
    changed = {k: v.replace('(36.0, 1.0, 6.0', '(35.5, 1.0, 6.0') for k, v in source.items()}
    assert source != changed
    assert review(source)['fingerprint'] == review(changed)['fingerprint']


def test_inline_core_and_transitive_helpers_never_become_inference_settings():
    source = {'ImageClassifier': '''class ImageClassifier:
    def core(self, x):
        return 0.5 * x
    def forward(self, x):
        logits = self.core(x)
        if self.training:
            return logits
        return 0.8 * logits
'''}
    wrapper = lambda methods, alias: {'helper_sha256': {'core': 'audited-helper'}}
    original = composition_key(source, _inference_key, wrapper)[0]
    changed = {'ImageClassifier': source['ImageClassifier'].replace('0.5 * x', '0.25 * x')}
    assert composition_key(changed, _inference_key, wrapper)[0] != original
    changed = {'ImageClassifier': source['ImageClassifier'].replace('logits = self.core(x)', 'logits = 0.5 * self.core(x)')}
    assert composition_key(changed, _inference_key, wrapper)[0] != original


def test_independently_admitted_weight_patterns_share_an_exact_core():
    refs = json.loads((Path(__file__).parents[1] / 'experiments/ontology_fashion_custom_references.json').read_text())['reviewed_profiles']
    before = refs[185]['program_parts']
    after = {k: v.replace('(3.0, 2.0, 2.0, 2.0, 2.0)', '(4.0, 3.0, 3.0, 3.0, 3.0)') for k, v in before.items()}
    assert before != after
    first = composition_key(before, _inference_key, _inference_reference, True)
    second = composition_key(after, _inference_key, _inference_reference, True)
    assert first and second and first[0] == second[0]
    result = review(after)
    assert result and result['fingerprint'] == refs[185]['fingerprint']
    for old, new in [('(4.0, 3.0, 3.0, 3.0, 3.0)', '(0.0, 3.0, 3.0, 3.0, 3.0)'),
                     ('dim=0).sum(dim=0)', 'dim=0).amax(dim=0)'),
                     ('kernel_size=3', 'kernel_size=5')]:
        changed = {k: v.replace(old, new) for k, v in after.items()}
        assert changed != after
        assert review(changed) is None


def test_scalar_skeleton_cannot_admit_wrapper_or_erase_training_or_family():
    source = {'ImageClassifier': '''class ImageClassifier:
    def core(self, x):
        return x
    def forward(self, x):
        logits = self.core(x)
        if self.training:
            return logits
        return 2.5 * logits
'''}
    admitted = lambda methods, alias: {'family_facts': {}}
    original = composition_key(source, _inference_key, admitted, True)[0]
    changed = {'ImageClassifier': source['ImageClassifier'].replace('2.5', '3.5')}
    assert composition_key(changed, _inference_key, admitted, True)[0] == original
    assert composition_key(changed, _inference_key, lambda *args: None, True) is None
    other_family = lambda methods, alias: {'family_facts': {'routing': 'new descriptor'}}
    assert composition_key(source, _inference_key, other_family, True)[0] != original
    changed = {'ImageClassifier': source['ImageClassifier'].replace('return logits\n', 'return 2.5 * logits\n')}
    assert composition_key(changed, _inference_key, admitted, True)[0] != original
    changed = {'ImageClassifier': source['ImageClassifier'].replace('logits = self.core(x)', 'logits = 0.5 * self.core(x)')}
    assert composition_key(changed, _inference_key, admitted, True)[0] != original
