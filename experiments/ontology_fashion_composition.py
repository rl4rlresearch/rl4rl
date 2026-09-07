"""Compose exact reviewed custom models with their admitted inference settings.

This does not infer a model from a class name or normalize constructor widths.
Every model AST node remains exact except whole methods already admitted by the
source-audited inference catalog, including its coefficient-relation guards.
The callbacks keep ownership of the inference contract in the vision reviewer.
"""
import ast
import copy
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path

from experiments.ontology_semantics import Specialize

REFERENCE_FILES = ('ontology_fashion_custom_references.json',
                   'ontology_fashion_lifecycle_references.json',
                   'ontology_fashion_standard_references.json')


def reference_revision():
    directory = Path(__file__).parent
    return tuple((name, (directory / name).stat().st_mtime_ns, (directory / name).stat().st_size)
                 for name in REFERENCE_FILES if (directory / name).exists())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class AdmittedScalarSettings(ast.NodeTransformer):
    """A reuse key after full independent wrapper admission, never admission."""
    def visit_Constant(self, node):
        if type(node.value) is float and math.isfinite(node.value) and node.value not in {0, 1, -1}:
            return ast.copy_location(ast.Constant(value='positive reviewed scalar' if node.value > 0 else 'negative reviewed scalar'), node)
        return node


def composition_key(parts, inference_key, inference_reference, shared_settings=False):
    trees = {name: ast.parse(source) for name, source in parts.items()}
    root = trees.get('ImageClassifier')
    if root is None or len(root.body) != 1 or not isinstance(root.body[0], ast.ClassDef):
        return None
    cls = root.body[0]
    methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
    if 'forward' not in methods:
        return None
    aliases = [name for name in ['_forward_once', '_predict', '_classify', '_forward_view', '_forward_logits']
               if name in methods and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                          and isinstance(n.func.value, ast.Name) and n.func.value.id == 'self'
                                          and n.func.attr == name for n in ast.walk(methods['forward']))]
    if len(aliases) > 1:
        return None
    alias = aliases[0] if aliases else None
    wrapper = inference_reference(methods, alias)
    if not wrapper:
        return None
    normalized_methods = {'forward', *wrapper.get('helper_sha256', {})}
    if alias in normalized_methods or '__init__' in normalized_methods:
        return None
    training_core = {}
    if alias is None or shared_settings:
        if not any(isinstance(n, ast.If) and ast.unparse(n.test) in {'self.training', 'not self.training'}
                   for n in ast.walk(methods['forward'])):
            return None
        training = Specialize({'self.training': True}).visit(copy.deepcopy(methods['forward']))
        for index, statement in enumerate(training.body):
            if isinstance(statement, ast.Return):
                training.body = training.body[:index + 1]
                break
        pending = [training]
        while pending:
            method = pending.pop()
            if method.name in training_core:
                continue
            training_core[method.name] = ast.dump(method, include_attributes=False)
            pending.extend(methods[n.attr] for n in ast.walk(method)
                           if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                           and n.value.id == 'self' and n.attr in methods and n.attr not in training_core)
    settings = {name: [str(n.value) for n in ast.walk(methods[name])
                       if isinstance(n, ast.Constant) and type(n.value) in (int, float)]
                for name in sorted(normalized_methods)}
    def method_key(method):
        return (ast.dump(AdmittedScalarSettings().visit(copy.deepcopy(method)), include_attributes=False)
                if shared_settings else inference_key(method))
    cls.body = [ast.Expr(value=ast.Constant(value='reviewed inference method: ' + method_key(m)))
                if isinstance(m, ast.FunctionDef) and m.name in normalized_methods else m
                for m in cls.body]
    key = _digest({'program': {name: ast.dump(tree, include_attributes=False) for name, tree in trees.items()},
                   'exact_training_core': training_core,
                   'wrapper_family_facts': wrapper.get('family_facts', {}),
                   'key_contract': 'independently-admitted-scalar-settings' if shared_settings else 'exact-admitted-wrapper'})
    return key, settings, wrapper


@lru_cache(maxsize=4)
def _templates(inference_key, inference_reference, revision, shared_settings=False):
    refs = []
    for name, _, _ in revision:
        refs.extend(json.loads(Path(__file__).with_name(name).read_text(encoding='utf-8'))['reviewed_profiles'])
    index, conflicts = {}, set()
    for ref in refs:
        match = composition_key(ref['program_parts'], inference_key, inference_reference, shared_settings)
        if match is None:
            continue
        key = match[0]
        previous = index.get(key)
        if previous and any(previous[k] != ref[k] for k in ('fingerprint', 'family_signature')):
            conflicts.add(key)
        else:
            index[key] = ref
    return {key: ref for key, ref in index.items() if key not in conflicts}


def reviewed_composition(parts, inference_key, inference_reference):
    match = composition_key(parts, inference_key, inference_reference)
    if match is None:
        return None
    key, settings, wrapper = match
    ref = _templates(inference_key, inference_reference, reference_revision()).get(key)
    shared_settings = False
    if ref is None:
        match = composition_key(parts, inference_key, inference_reference, True)
        if match is None:
            return None
        key, settings, wrapper = match
        ref = _templates(inference_key, inference_reference, reference_revision(), True).get(key)
        shared_settings = True
    if ref is None:
        return None
    if shared_settings:
        original = composition_key(ref['program_parts'], inference_key, inference_reference, True)[1]
        changed = {float(a) for name in settings for a, b in zip(original[name], settings[name]) if a != b}
        mentioned = {float(value) for text in ref['fingerprint'].values() for value in
                     re.findall(r'(?<![\w.])-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?(?![\w.])', text, re.I)}
        if changed & mentioned:
            # Do not copy a literal coefficient claim after that value changes.
            return None
    result = {k: copy.deepcopy(ref[k]) for k in
              ('fingerprint', 'family_signature', 'family_label', 'training', 'inference', 'settings', 'notes', 'evidence')
              if k in ref}
    result['inference'] = copy.deepcopy(wrapper['inference'])
    result.setdefault('settings', {})['inference_numeric_literals'] = settings
    result['notes'] += ' The complete custom model matches this direct review; its full inference methods match the guarded inference catalog, with numeric settings recorded separately.'
    result['evidence'].append({'kind': 'exact custom model with source-audited inference settings',
                               'composition_sha256': key, 'program_sha256': _digest(parts),
                               'reference_source_sha256': ref['source_sha256'],
                               'forward_template_sha256': wrapper['forward_sha256'],
                               'helper_template_sha256': wrapper.get('helper_sha256', {}),
                               'composition_contract': 'independently admitted scalar settings' if shared_settings else 'exact admitted wrapper',
                               'settings': settings})
    return result
