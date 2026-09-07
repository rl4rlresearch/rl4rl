"""Closed parameter slots in directly reviewed Fashion inference programs.

Only the positive literal coefficient multiplying the already traced confidence
margin, and a positive literal temperature after fusion, can vary. All model
methods, inputs, axes, branches and other operators must match a direct review.
"""
import ast
import copy
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path


def positive_literal(node):
    return (isinstance(node, ast.Constant) and type(node.value) in (int, float)
            and math.isfinite(node.value) and node.value > 0)


def template_parts(parts):
    normalized, settings = {}, {}
    for name, source in parts.items():
        tree = ast.parse(source)
        if name == 'ImageClassifier':
            for cls in tree.body:
                if not isinstance(cls, ast.ClassDef):
                    continue
                for method in cls.body:
                    if not isinstance(method, ast.FunctionDef) or method.name != 'forward':
                        continue
                    for node in ast.walk(method):
                        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'F'
                                and node.func.attr == 'softmax' and node.args):
                            argument = node.args[0]
                            if (isinstance(argument, ast.BinOp) and isinstance(argument.op, ast.Mult)
                                    and positive_literal(argument.left) and isinstance(argument.right, ast.Name)
                                    and argument.right.id == 'margins'):
                                settings['confidence_margin_coefficient'] = str(argument.left.value)
                                argument.left = ast.Constant(value='positive confidence-margin coefficient')
                        if isinstance(node, ast.Return):
                            expression = node.value
                            if (isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Div)
                                    and isinstance(expression.left, ast.Name)
                                    and expression.left.id == 'fused_log_probabilities'
                                    and positive_literal(expression.right)):
                                settings['output_temperature'] = str(expression.right.value)
                                expression.right = ast.Constant(value='positive fused-output temperature')
        normalized[name] = ast.dump(tree, include_attributes=False)
    key = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
    return key, settings


@lru_cache(maxsize=1)
def templates():
    path = Path(__file__).with_name('ontology_fashion_custom_references.json')
    refs = json.loads(path.read_text(encoding='utf-8'))['reviewed_profiles']
    index, conflicts = {}, set()
    for ref in refs:
        key, settings = template_parts(ref['program_parts'])
        if not settings:
            continue
        if key in index and any(index[key][k] != ref[k] for k in ('fingerprint', 'family_signature')):
            conflicts.add(key)
        else:
            index[key] = ref
    return {key: ref for key, ref in index.items() if key not in conflicts}


def reviewed_template(parts):
    key, settings = template_parts(parts)
    ref = templates().get(key)
    if ref is None or not settings:
        return None
    result = {k: copy.deepcopy(ref[k]) for k in
              ('fingerprint', 'family_signature', 'family_label', 'training', 'inference', 'notes', 'evidence') if k in ref}
    result['settings'] = settings
    result['notes'] += ' Every model AST node matches this direct review except explicitly bounded positive confidence-margin/output-temperature literals.'
    result['evidence'].append({'kind': 'closed reviewed scalar template', 'template_sha256': key,
                               'settings': settings, 'reference_source_sha256': ref['source_sha256']})
    return result
