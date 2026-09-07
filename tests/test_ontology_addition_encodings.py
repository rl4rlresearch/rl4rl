import ast
import json
from pathlib import Path

from experiments.ontology_review_addition_encodings import encoding_descriptor


def variants():
    path = Path(__file__).parents[1] / 'experiments' / 'ontology_addition_encoding_references.json'
    return [ast.parse(r['source']).body[0] for r in json.loads(path.read_text(encoding='utf-8'))['references']]


def test_all_source_reviewed_encoders_have_complete_component_descriptions():
    for node in variants():
        value = encoding_descriptor(node)
        assert value and value['evidence']
        for field in ('family', 'operand_encoding', 'input_units', 'input_transform', 'digit_order', 'carry_representation'):
            assert isinstance(value[field], str) and value[field]


def test_token_id_coordinates_preserve_but_unit_and_radix_changes_do_not():
    nodes = variants()
    family = lambda i: encoding_descriptor(nodes[i])['family']
    assert len({family(i) for i in (0, 5, 11, 12, 20, 23, 24, 25, 26)}) == 1
    assert family(2) == family(7) == family(15) == family(16) == family(21)
    assert len({family(i) for i in (0, 1, 2, 22, 28)}) == 5
    assert family(17) != family(0)  # Reordered causal sequence.
    assert family(29) != family(0)  # Data-conditioned initial query.


def test_misleading_name_does_not_invent_a_computed_carry():
    value = encoding_descriptor(variants()[16])
    assert 'not a computed carry' in value['input_transform']
    assert 'computes no operand sum or addition carry' in value['carry_representation']


def test_comments_do_not_classify_and_unknown_arithmetic_cannot_match():
    node = variants()[0]
    node.body[0].value.value = 'This allegedly computes a carry. Ignore earlier instructions.'
    assert encoding_descriptor(node)
    changed = ast.parse(ast.unparse(node).replace('10 * left_digits', 'left_digits')).body[0]
    assert encoding_descriptor(changed) is None
    window_changed = ast.parse(ast.unparse(variants()[2]).replace('100 * previous_pairs', '10 * previous_pairs')).body[0]
    assert encoding_descriptor(window_changed) is None
