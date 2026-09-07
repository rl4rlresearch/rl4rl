import json
from pathlib import Path

from experiments.ontology_fashion_templates import reviewed_template


def parts():
    file = Path(__file__).parents[1] / 'experiments' / 'ontology_fashion_custom_references.json'
    return json.loads(file.read_text(encoding='utf-8'))['reviewed_profiles'][18]['program_parts']


def test_positive_coefficient_sweeps_retain_a_source_reviewed_family():
    source = parts()
    changed = {k: v.replace('0.08729376494884492 * margins', '0.4321 * margins').replace('/ 0.800713', '/ 0.83') for k, v in source.items()}
    result = reviewed_template(changed)
    assert result
    assert result['settings'] == {'confidence_margin_coefficient': '0.4321', 'output_temperature': '0.83'}
    assert result['family_signature'] == reviewed_template(source)['family_signature']


def test_disabling_input_routing_or_altering_signal_math_cannot_reuse_template():
    source = parts()
    for old, new in [('0.08729376494884492 * margins', '0.0 * margins'),
                     ('0.08729376494884492 * margins', 'coefficient * margins'),
                     ('0.08729376494884492 * margins', '0.2 / margins'),
                     ('topk(2, dim=2)', 'topk(2, dim=1)'),
                     ('torch.sigmoid(channel_gate)', 'torch.sin(channel_gate)')]:
        changed = {k: v.replace(old, new) for k, v in source.items()}
        assert changed != source
        assert reviewed_template(changed) is None
