"""Source-audited ontology rules for the Fashion-MNIST and nanoGPT campaigns.

The campaign rubric treats widths, depths, affine biases, conventional pointwise
activations/normalization, fixed additive residuals, and training/TTA settings as
preserving. A new signal-dependent gate, statistic/history basis, nonlocal
relation, persistent state, prototype/covariance representation, or learned
cross-view aggregation is a boundary. Fingerprint details still record routine
changes. Handoff page 3 explicitly treats weight tying and projection
simplification as preserving: grouped-query KV reuse and dense/grouped/depthwise
local convolution connectivity therefore remain fingerprint/settings details.
Unrecognized programs return None; absence of a pattern is never a
complete fingerprint. Candidate Python is parsed, never imported or executed.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from functools import lru_cache
from fractions import Fraction
from pathlib import Path

from experiments.ontology_semantics import program, constant, UNKNOWN, Specialize, assign_constants
from experiments.ontology_seed_rubric import seed_fingerprint
from experiments.ontology_transition_review import NumericSettings
from experiments.review_ontology_sources import dotted
from experiments.ontology_fashion_templates import reviewed_template
from experiments.ontology_fashion_composition import reviewed_composition

VERSION = "vision-lm-semantic-v1"
FASHION = "openevolve_v21_fashion_mnist"
NANO = "openevolve_v21_nanogpt"


class Unsupported(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=4)
def _json_revision(path,mtime,size):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _read_json(path):
    st=path.stat()
    return _json_revision(str(path),st.st_mtime_ns,st.st_size)


def _fashion_reference(sources, parts=None):
    sha=digest(sources) if parts is None else digest(parts)
    field='source_sha256' if parts is None else 'program_sha256'
    # Candidate batches are deliberately staged out-of-band until their
    # source-complete fingerprints have passed the reference suite.
    for filename in ('ontology_fashion_custom_references.json',
                     'ontology_fashion_lifecycle_references.json', 'ontology_fashion_standard_references.json'):
        path=Path(__file__).with_name(filename)
        if not path.exists():continue
        for ref in _read_json(path).get('reviewed_profiles',[]):
            if ref.get(field)==sha:
                return {k:copy.deepcopy(ref[k]) for k in ['fingerprint','family_signature','family_label','training','inference','settings','notes','evidence'] if k in ref}
    return None


def _staged_fashion_candidate_pair(before_sources, after_sources):
    """Return a finite candidate decision only for its recorded exact parent.

    This catalog is intentionally absent from `_fashion_reference`: a candidate
    must never become a reusable single-source profile.  Its declaration is
    usable solely while adjudicating the one source-hash-bound parent-child
    transition that was directly read.
    """
    if not before_sources or not after_sources:
        return None
    child_hash, parent_hash = digest(after_sources), digest(before_sources)
    for filename in ("ontology_fashion_candidate_references_v1.json", "ontology_fashion_candidate_references_v2.json",
                     "ontology_fashion_candidate_references_v3_unresolved.json", "ontology_fashion_candidate_references_v4_b04c1.json", "ontology_fashion_candidate_references_v5_b05c1_readouts.json", "ontology_fashion_candidate_references_v6_b02c3_crop_fusion.json", "ontology_fashion_candidate_references_v7_b02c3_powered_fusion.json", "ontology_fashion_candidate_references_v8_b02c3_log_calibration.json", "ontology_fashion_candidate_references_v9_b02c3_orientation_weights.json", "ontology_fashion_candidate_references_v10_b02c3_view_balance.json"):
        path = Path(__file__).with_name(filename)
        if not path.exists():
            continue
        for ref in _read_json(path).get("reviewed_profiles", []):
            if ref.get("source_sha256") != child_hash or ref.get("parent_source_sha256") != parent_hash:
                continue
            witness = {
                "kind": "exact staged candidate-parent transition declaration",
                "source_sha256": child_hash,
                "parent_source_sha256": parent_hash,
                "transition_classification": ref["transition_classification"],
                "changed_components": ref["changed_components"],
            }
            return {"classification": ref["transition_classification"], "fingerprint": copy.deepcopy(ref["fingerprint"]),
                "fingerprint_complete": True, "family_signature": copy.deepcopy(ref["family_signature"]),
                "changed_components": copy.deepcopy(ref["changed_components"]),
                "notes": ref["notes"], "evidence": copy.deepcopy(ref["evidence"]) + [witness],
                "inference": copy.deepcopy(ref.get("inference", {})), "family_label": ref.get("family_label"),
                "training": copy.deepcopy(ref.get("training", {})), "settings": copy.deepcopy(ref.get("settings", {})),
                "review_method": VERSION + "+staged-exact-parent-pair", "source_sha256": child_hash,
                "parent_source_sha256": parent_hash}
    return None


def _key(node):
    return ast.dump(NumericSettings().visit(copy.deepcopy(node)), include_attributes=False)


class InferenceSettings(NumericSettings):
    """Only float coefficients vary in audited inference templates.

    Integer dimensions, partition counts, indices, top-k ranks and control
    thresholds stay exact. This is deliberately narrower than model widths.
    """
    def __init__(self):
        self.float_ids = {}

    def visit_Constant(self, node):
        if type(node.value) is float and node.value not in {0, 1, -1} and math.isfinite(node.value):
            # Two independently tunable mixtures must not become identical by
            # collapsing different literals to one undifferentiated marker.
            index = self.float_ids.setdefault(node.value, len(self.float_ids))
            return ast.copy_location(ast.Constant(value=('positive' if node.value > 0 else 'negative') + ' float setting ' + str(index)), node)
        return node

    def visit_Call(self, node):
        return self.generic_visit(node)


def _inference_key(node):
    tree = ast.dump(InferenceSettings().visit(copy.deepcopy(node)), include_attributes=False)
    return tree + '\nlinear coefficient relations: ' + json.dumps(_inference_linear_relations(node), sort_keys=True)


def _inference_linear_relations(node):
    """Bind cancellations and proportional mixtures in the reviewed template.

    Positive scalar multiplication preserves an argmax. If two formerly
    different mixture formulas become proportional, an argmax-dependent route
    can disappear even though every literal remains nonzero. This analysis is
    only an additional admission guard; it never assigns a new family label.
    """
    def form(value):
        if isinstance(value, ast.Constant) and type(value.value) in (int, float):
            if not math.isfinite(value.value):
                return None
            return {'#constant': Fraction(str(value.value))}
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
            terms = form(value.operand)
            return None if terms is None else {k: v * (-1 if isinstance(value.op, ast.USub) else 1) for k, v in terms.items()}
        if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = form(value.left), form(value.right)
            if left is None or right is None:
                return None
            if isinstance(value.op, (ast.Add, ast.Sub)):
                sign = -1 if isinstance(value.op, ast.Sub) else 1
                return {k: left.get(k, 0) + sign * right.get(k, 0) for k in left.keys() | right.keys()}
            if set(right) <= {'#constant'}:
                scalar = right.get('#constant', 0)
                if isinstance(value.op, ast.Div):
                    return None if not scalar else {k: v / scalar for k, v in left.items()}
                return {k: v * scalar for k, v in left.items()}
            if isinstance(value.op, ast.Mult) and set(left) <= {'#constant'}:
                return {k: v * left.get('#constant', 0) for k, v in right.items()}
            return None
        return {ast.dump(value, include_attributes=False): Fraction(1)}

    forms = []
    for statement in ast.walk(node):
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.BinOp):
            terms = form(statement.value)
            if terms is not None:
                forms.append(terms)
    zeros = [(i, sorted(k for k, v in terms.items() if not v)) for i, terms in enumerate(forms)]
    proportional = []
    for i, left in enumerate(forms):
        a = {k: v for k, v in left.items() if v}
        for j in range(i):
            b = {k: v for k, v in forms[j].items() if v}
            if a and a.keys() == b.keys():
                key = next(iter(a))
                ratio = a[key] / b[key]
                if all(a[k] == ratio * b[k] for k in a):
                    proportional.append((j, i, 'positive' if ratio > 0 else 'negative'))
    return {'zero_coefficients': zeros, 'proportional_mixtures': proportional}


def _inference_reference(methods, alias):
    path = Path(__file__).with_name('ontology_fashion_inference_references.json')
    if not path.exists():
        return None
    key = digest(_inference_key(methods['forward']))
    for ref in _read_json(path).get('reviewed_wrappers', []):
        if ref['forward_sha256'] != key or ref['model_alias'] != alias:
            continue
        if any(name not in methods or digest(_inference_key(methods[name])) != sha for name, sha in ref.get('helper_sha256', {}).items()):
            continue
        return copy.deepcopy(ref)
    return None


def _stmt(node):
    return {"line": node.lineno, "code": ast.unparse(node).splitlines()[0][:300]}


def _model_reached(defs, roots):
    reached = set(roots)
    while True:
        previous = set(reached)
        for name in previous:
            node = copy.deepcopy(defs[name])
            if isinstance(node, ast.ClassDef):
                node.body = [m for m in node.body if not isinstance(m, ast.FunctionDef) or m.name not in {"setup_optimizer", "num_scaling_params", "estimate_flops"}]
            reached.update(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in defs)
        if previous == reached:
            return reached


PLAIN_LAYERS = {
    "Conv2d", "Linear", "Sequential", "ModuleList", "Identity", "Flatten",
    "BatchNorm1d", "BatchNorm2d", "LayerNorm", "GroupNorm", "InstanceNorm2d",
    "GELU", "ReLU", "ReLU6", "SiLU", "LeakyReLU", "PReLU", "ELU", "Tanh", "Sigmoid",
    "Dropout", "Dropout2d", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "AdaptiveMaxPool2d",
}
ACTS = {"gelu", "relu", "relu6", "silu", "leaky_relu", "elu", "tanh", "sigmoid"}
NORMS = {"BatchNorm1d", "BatchNorm2d", "LayerNorm", "GroupNorm", "InstanceNorm2d"}
INIT_CALLS = {"super", ".__init__", "nn.init.zeros_", "nn.init.ones_", "nn.init.constant_", "nn.init.normal_", "nn.init.uniform_", "nn.init.kaiming_normal_", "nn.init.kaiming_uniform_", "nn.init.trunc_normal_", "max", "min", "int", "float"}


def _module_aliases(method, modules):
    """Single-assignment local aliases retain the exact declared module object."""
    counts = {}
    for node in ast.walk(method):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            counts[node.id] = counts.get(node.id, 0) + 1
    return {node.targets[0].id: dotted(node.value) for node in method.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and counts.get(node.targets[0].id) == 1 and dotted(node.value) in modules}


def _storage_conversion(node):
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in {'to', 'contiguous'}:
        return False
    float_types = {'torch.float16', 'torch.float32', 'torch.float64', 'torch.bfloat16', 'torch.float', 'torch.double', 'torch.half'}
    layouts = {'torch.channels_last', 'torch.contiguous_format', 'torch.preserve_format'}
    if len(node.args) > 1 or (node.args and (node.func.attr != 'to' or dotted(node.args[0]) not in float_types)):
        return False
    for keyword in node.keywords:
        if keyword.arg == 'memory_format' and dotted(keyword.value) in layouts:
            continue
        if keyword.arg == 'dtype' and node.func.attr == 'to' and dotted(keyword.value) in float_types:
            continue
        if keyword.arg in {'copy', 'non_blocking'} and node.func.attr == 'to' and isinstance(keyword.value, ast.Constant) and type(keyword.value.value) is bool:
            continue
        return False
    return True


def _literal_environment(method):
    """Only single-assignment literal locals can become tensor coefficients."""
    writes = {}
    for n in ast.walk(method):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for target in targets:
                for t in ast.walk(target):
                    if isinstance(t, ast.Name):
                        writes.setdefault(t.id, []).append(n.value if isinstance(target, ast.Name) and not isinstance(n, ast.AugAssign) else None)
        if isinstance(n, (ast.For, ast.comprehension)):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    writes.setdefault(t.id, []).append(None)
    env = {}
    for _ in range(8):
        old = dict(env)
        for name, values in writes.items():
            if len(values) == 1:
                value = constant(values[0], env)
                if value is not UNKNOWN:
                    env[name] = value
        if old == env:
            break
    return env


def _tta_fixed_control(method, constants):
    """Reject signal-dependent branch, crop, and loop coordinates in a wrapper.

    Shape metadata and indices of fixed view collections are configuration;
    tensor contents are not. A None sentinel only initializes an accumulator or
    selects the availability of an EMA snapshot, rather than routing on values.
    """
    structural = set()
    snapshots = set()
    writes = {}
    for n in ast.walk(method):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            writes[n.id] = writes.get(n.id, 0) + 1

    def fixed(expr):
        if expr is None or constant(expr, constants) is not UNKNOWN:
            return True
        if isinstance(expr, ast.Name):
            return expr.id in structural
        if isinstance(expr, ast.Attribute):
            return expr.attr in {'shape', 'ndim'}
        if isinstance(expr, (ast.Tuple, ast.List)):
            return all(fixed(v) for v in expr.elts)
        if isinstance(expr, ast.Slice):
            return all(fixed(v) for v in (expr.lower, expr.upper, expr.step))
        if isinstance(expr, ast.Subscript):
            return fixed(expr.value) and fixed(expr.slice)
        if isinstance(expr, ast.UnaryOp):
            return fixed(expr.operand)
        if isinstance(expr, ast.BinOp):
            return fixed(expr.left) and fixed(expr.right)
        if isinstance(expr, ast.Call):
            name = dotted(expr.func)
            if name == 'len':
                return True  # collection/tensor shape, never tensor contents
            if name in {'int', 'float', 'range'}:
                return all(fixed(v) for v in expr.args)
        return False

    # Fixed-point propagation covers unpacked shape metadata and literal offsets.
    for _ in range(8):
        before = set(structural)
        for n in ast.walk(method):
            if isinstance(n, ast.Assign):
                if fixed(n.value):
                    structural.update(t.id for target in n.targets for t in ast.walk(target) if isinstance(t, ast.Name) and writes.get(t.id) == 1)
                if isinstance(n.value, ast.Call) and dotted(n.value.func) == 'getattr' and len(n.value.args) >= 2 and isinstance(n.value.args[1], ast.Constant) and n.value.args[1].value in {'_ema_state', '_ema_states'}:
                    snapshots.update(t.id for target in n.targets for t in ast.walk(target) if isinstance(t, ast.Name) and writes.get(t.id) == 1)
            if isinstance(n, (ast.For, ast.comprehension)):
                if fixed(n.iter):
                    structural.update(t.id for t in ast.walk(n.target) if isinstance(t, ast.Name) and writes[t.id] == 1)
                elif isinstance(n.iter, ast.Call) and dotted(n.iter.func) == 'enumerate' and isinstance(n.target, (ast.Tuple, ast.List)) and isinstance(n.target.elts[0], ast.Name):
                    if writes[n.target.elts[0].id] == 1:
                        structural.add(n.target.elts[0].id)
        if before == structural:
            break

    def branch(expr):
        if dotted(expr) == 'self.training' or fixed(expr):
            return True
        if isinstance(expr, ast.Name) and expr.id in snapshots:
            return True
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            return branch(expr.operand)
        if isinstance(expr, ast.BoolOp):
            return all(branch(v) for v in expr.values)
        if isinstance(expr, ast.Compare):
            if len(expr.ops) == 1 and isinstance(expr.ops[0], (ast.Is, ast.IsNot)) and isinstance(expr.comparators[0], ast.Constant) and expr.comparators[0].value is None:
                return isinstance(expr.left, ast.Name)
            return fixed(expr.left) and all(fixed(v) for v in expr.comparators)
        return False

    for n in ast.walk(method):
        if isinstance(n, (ast.If, ast.IfExp)) and not branch(n.test):
            return False
        if isinstance(n, ast.comprehension) and any(not branch(test) for test in n.ifs):
            return False
        if isinstance(n, ast.Subscript) and not fixed(n.slice):
            return False
        if isinstance(n, ast.Call) and dotted(n.func) == 'range' and not fixed(n):
            return False
    return True


def _tta_wrapper(method, model_calls=None):
    """Closed fixed-view probability/logit averaging wrapper, including EMA.

    This grammar admits no learned gate, parameterized readout, feature-summary
    reduction or signal-dependent sampling. EMA access only swaps snapshots for
    the same `_forward_once` model, then restores the original tensors.
    """
    simple = {"self._forward_once", "self.named_parameters", "self.named_buffers", "dict", "list", "tuple", "len", "enumerate", "range", "float", "int", "zip", "math.log", "math.exp", "torch.no_grad", "F.pad", "torch.flip", "F.log_softmax", "torch.logsumexp", "torch.logaddexp", "torch.stack", "torch.cat"}
    simple.update(model_calls or [])
    simple.update(n.name for n in ast.walk(method) if isinstance(n, ast.FunctionDef) and n is not method)
    constants = _literal_environment(method)
    if not _tta_fixed_control(method, constants):
        return False
    coordinate = {"flip", "append", "extend", "items", "update", "detach", "clone", "copy_", "chunk", "split", "contiguous", "reshape", "view", "mean", "sum"}
    for n in ast.walk(method):
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.While, ast.Lambda, ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr, ast.Delete)):
            return False
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and dotted(n).startswith('self.'):
            return False
        if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store):
            return False
        if isinstance(n, ast.Call):
            name = dotted(n.func)
            if name in simple:
                continue
            if name == "getattr" and len(n.args)>=2 and dotted(n.args[0])=="self" and isinstance(n.args[1],ast.Constant) and n.args[1].value in {"_ema_state", "_ema_states"}:
                continue
            if isinstance(n.func,ast.Attribute) and n.func.attr in coordinate and not dotted(n.func.value).startswith("self"):
                continue
            return False
        if isinstance(n,ast.Attribute) and dotted(n).startswith("self.") and dotted(n) not in {"self.training", "self._forward_once", "self.named_parameters", "self.named_buffers"} | set(model_calls or []):
            return False
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Mult,ast.Div)):
            # Logit coefficients must be literal/configuration arithmetic.
            def scalar(x):
                if constant(x, constants) is not UNKNOWN:
                    return True
                return isinstance(x, ast.Call) and (dotted(x.func) == 'len' or dotted(x.func) in {'math.log', 'math.exp'} and all(scalar(a) for a in x.args))
            if not scalar(n.left) and not scalar(n.right):
                return False
            if isinstance(n.op, ast.Div) and not scalar(n.right):
                return False
            if isinstance(n.op, ast.Div) and constant(n.right, constants) == 0:
                return False
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Pow, ast.MatMult, ast.BitAnd, ast.BitOr, ast.BitXor)):
            return False
    return True


def _terminal_global_summary(method, modules):
    """Trace the descriptor entering a terminal classifier call.

    This narrow trace recognizes global-pool -> flatten -> optional feature
    concatenation. It does not infer a global readout from unrelated local pools.
    """
    bindings = {}
    repeated = set()
    for n in method.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if name in bindings:
                repeated.add(name)
            bindings[name] = n.value

    def resolve(expr, seen):
        if isinstance(expr, ast.Name) and expr.id in bindings and expr.id not in repeated and expr.id not in seen:
            return resolve(bindings[expr.id], seen | {expr.id})
        return expr

    def summary(expr, seen=frozenset()):
        expr = resolve(expr, seen)
        if not isinstance(expr, ast.Call):
            return None
        called = dotted(expr.func)
        if called in {'F.adaptive_avg_pool2d', 'F.adaptive_max_pool2d'}:
            size = expr.args[1] if len(expr.args) > 1 else next((k.value for k in expr.keywords if k.arg == 'output_size'), None)
            if constant(size, {}) in (1, (1, 1), [1, 1]):
                return {'mean' if 'avg' in called else 'maximum'}
        if isinstance(expr.func, ast.Attribute) and expr.func.attr == 'flatten' and len(expr.args) == 1 and constant(expr.args[0], {}) == 1:
            return summary(expr.func.value, seen)
        if called == 'torch.cat' and expr.args and isinstance(expr.args[0], (ast.Tuple, ast.List)):
            parts = [summary(value, seen) for value in expr.args[0].elts]
            if parts and all(parts):
                return set().union(*parts)
        return None

    for n in reversed(method.body):
        if isinstance(n, ast.Return):
            value = resolve(n.value, set())
            if isinstance(value, ast.Call) and dotted(value.func) in modules and len(value.args) == 1:
                return summary(value.args[0])
    return None


def _fashion_model(sources):
    direct=_fashion_reference(sources)
    if direct:return direct
    defs, files, classes, reached, roots, envs, glob, decisions, trees = program(sources)
    reached = _model_reached(defs, roots)
    parts={name:ast.unparse(defs[name]) for name in sorted(reached)}
    direct=_fashion_reference(sources,parts)
    if direct:return direct
    direct=reviewed_template(parts)
    if direct:return direct
    direct=reviewed_composition(parts, _inference_key, _inference_reference)
    if direct:return direct
    root = next(iter(roots))
    if root != "ImageClassifier" or reached - classes:
        raise Unsupported("Extra model helper needs semantic review")
    constructors, methods, modules, primitive_calls = {}, {}, {}, []
    for name in reached:
        cl = defs[name]
        methods[name] = {m.name: m for m in cl.body if isinstance(m, ast.FunctionDef)}
        if [dotted(base) for base in cl.bases] != ['nn.Module']:
            raise Unsupported('Nonstandard model inheritance needs separate review')
        if set(methods[name]) & {'__call__', '_call_impl', '_wrapped_call_impl', '__getattribute__', '__getattr__', '__setattr__', 'train', 'eval', '_apply'}:
            raise Unsupported('Model execution override needs separate review')
        if any(not isinstance(m, ast.FunctionDef) for m in cl.body):
            raise Unsupported("Class-level behavior outside methods")
        if cl.decorator_list or any(m.decorator_list for m in methods[name].values()):
            raise Unsupported("Model decorators need separate review")
        ctor = methods[name].get("__init__")
        constructors[name] = ctor
        modules[name] = set()
        if ctor:
            for n in ast.walk(ctor):
                if isinstance(n, ast.Assign):
                    for target in n.targets:
                        if dotted(target).startswith('self.') and dotted(target)[5:] in methods[name]:
                            raise Unsupported('Model method rebinding needs separate review')
                        if isinstance(target, ast.Subscript):
                            raise Unsupported("Constructor indexed state assignment needs separate review")
                        if dotted(target).startswith('self.') and dotted(target).count('.') > 1:
                            raise Unsupported("Parameter rebinding or sharing needs separate review")
                        if dotted(target).startswith("self.") and isinstance(n.value, ast.Call):
                            modules[name].add(dotted(target))
                if isinstance(n, ast.Call):
                    called = dotted(n.func)
                    leaf = called.split(".")[-1]
                    if called.startswith("nn.") and leaf in PLAIN_LAYERS:
                        primitive_calls.append((name, n, leaf))
                    elif called in reached or called in INIT_CALLS:
                        pass
                    else:
                        raise Unsupported("Nonstandard constructor operation: " + called)
                if isinstance(n, (ast.For, ast.While, ast.Try, ast.With, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                    raise Unsupported("Constructor control flow needs separate review")
    # Only standard module execution and fixed additive residuals are admitted.
    # Multiplicative feature interactions cannot slip through this grammar.
    evidence = []
    visited = set()
    residual = False
    concatenated = False
    flattened = False
    aggregations = set()
    functional_acts = set()
    functional_pools = set()
    functional_dropout = set()
    storage_conversions = set()
    inference = set()
    inference_reference = None
    root_readout_summaries = set()

    def method(name, member, wrapper=False, override=None):
        nonlocal residual, flattened, concatenated
        ident = (name, member)
        if ident in visited:
            return
        visited.add(ident)
        m = override or methods[name].get(member)
        if m is None:
            raise Unsupported("Unresolved method " + str(ident))
        evidence.append(dict(_stmt(m), file=files[name], scope=name + "." + member))
        if name == root:
            root_readout_summaries.update(_terminal_global_summary(m, modules[name]) or set())
        # Constructor-local names can be shadowed by tensor-valued method locals.
        arithmetic_env = dict(envs[name])
        for n in ast.walk(m):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                arithmetic_env.pop(n.id, None)
        for arg in m.args.args + m.args.kwonlyargs:
            arithmetic_env.pop(arg.arg, None)
        arithmetic_env.update(_literal_environment(m))
        module_aliases = _module_aliases(m, modules[name])
        for n in ast.walk(m):
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and dotted(n).startswith('self.'):
                raise Unsupported("Model state mutation needs separate review")
            if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store):
                raise Unsupported("Indexed model state mutation needs separate review")
            if isinstance(n, (ast.While, ast.Try, ast.With, ast.Lambda, ast.Yield, ast.YieldFrom, ast.Await, ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.NamedExpr, ast.Delete)):
                raise Unsupported("Runtime control flow needs separate review")
            if isinstance(n, ast.If):
                if ast.unparse(n.test) not in {"self.training", "not self.training"}:
                    raise Unsupported("Runtime branch needs separate review")
                inference.add("training/evaluation branch")
            if isinstance(n, ast.For):
                raise Unsupported("Runtime loop needs separate review")
            if isinstance(n, ast.BinOp):
                if isinstance(n.op, ast.Add):
                    residual = True
                elif isinstance(n.op, (ast.Mult, ast.Div)):
                    # Only fixed literal coefficients on a tensor are routine.
                    right=constant(n.right,arithmetic_env)
                    if isinstance(n.op,ast.Div) and (right is UNKNOWN or right==0):
                        raise Unsupported("Reciprocal feature transform or invalid denominator")
                    if right is UNKNOWN and constant(n.left, arithmetic_env) is UNKNOWN:
                        raise Unsupported("Product of nonconstant terms")
                else:
                    raise Unsupported("Non-affine feature arithmetic")
            if isinstance(n, ast.Subscript):
                if not dotted(n.value).startswith("self.") and dotted(n.value) not in module_aliases:
                    raise Unsupported("Feature selection needs separate review")
                if not isinstance(n.slice, ast.Constant) or type(n.slice.value) is not int:
                    raise Unsupported("Dynamic module selection")
            if not isinstance(n, ast.Call):
                continue
            called = dotted(n.func)
            leaf = called.split(".")[-1]
            if isinstance(n.func, ast.Subscript):
                base = dotted(n.func.value)
                base = module_aliases.get(base, base)
                if base not in modules[name]:
                    raise Unsupported("Unknown module list")
                continue
            if called in module_aliases:
                continue
            if called.startswith("self."):
                if called in modules[name]:
                    continue
                if called[5:] in methods[name]:
                    method(name, called[5:])
                    continue
                raise Unsupported("Unknown self call " + called)
            if called.startswith(("F.", "torch.")) and leaf in ACTS:
                functional_acts.add(leaf)
                continue
            if called in {'F.dropout', 'F.dropout2d'}:
                probability = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == 'p'), ast.Constant(value=0.5))
                training = n.args[2] if len(n.args) > 2 else next((k.value for k in n.keywords if k.arg == 'training'), ast.Constant(value=True))
                p = constant(probability, arithmetic_env)
                if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 or ast.unparse(training) not in {'self.training', 'False'}:
                    raise Unsupported('Nonstandard dropout control needs separate review')
                functional_dropout.add(called + ' with fixed p and training=' + ast.unparse(training))
                continue
            if _storage_conversion(n):
                storage_conversions.add(ast.unparse(n))
                continue
            if called.startswith(("F.", "torch.")) and leaf in {"max_pool2d", "avg_pool2d", "adaptive_avg_pool2d", "adaptive_max_pool2d"}:
                functional_pools.add(leaf)
                continue
            if called == 'torch.cat':
                dim = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == 'dim'), None)
                if not n.args or not isinstance(n.args[0], (ast.Tuple, ast.List)) or constant(dim, arithmetic_env) != 1:
                    raise Unsupported('Nonstandard concatenation axis or collection')
                concatenated = True
                continue
            if leaf == "flatten":
                # torch.flatten(x, 1) and x.flatten(1) have the same layout.
                # Keep batch-axis merging and partial flattening outside this
                # ordinary flattened-readout rule.
                arguments = n.args[1:] if called == 'torch.flatten' else n.args
                keywords = {k.arg: k.value for k in n.keywords}
                start = arguments[0] if arguments else keywords.get('start_dim', ast.Constant(value=0))
                end = arguments[1] if len(arguments) > 1 else keywords.get('end_dim', ast.Constant(value=-1))
                if (len(arguments) > 2 or set(keywords) - {'start_dim', 'end_dim'}
                        or constant(start, arithmetic_env) != 1 or constant(end, arithmetic_env) != -1):
                    raise Unsupported("Nonstandard flatten axes")
                flattened = True
                continue
            if leaf == "flip":
                if not wrapper:
                    raise Unsupported("Feature-view transformation inside model")
                inference.add("fixed horizontal-flip ensemble")
                continue
            raise Unsupported("Nonstandard runtime operation: " + called)

    # The simplest TTA wrappers are checked in their entirety. More elaborate
    # EMA/crop wrappers are intentionally not silently dropped.
    aliases=[name for name in ['_forward_once','_predict','_classify','_forward_view','_forward_logits'] if name in methods[root] and any(isinstance(n,ast.Call) and dotted(n.func)=='self.'+name for n in ast.walk(methods[root]['forward']))]
    if len(aliases) <= 1:
        inference_reference = _inference_reference(methods[root], aliases[0] if aliases else None)
    if inference_reference:
        if inference_reference.get('composition_only'):
            raise Unsupported('This inference wrapper requires an exact reviewed custom constructor and core')
        if aliases:
            method(root, aliases[0])
        else:
            # The entire original inference method and every helper were matched
            # before separating its already-reviewed training computation.
            training = Specialize({"self.training": True}).visit(copy.deepcopy(methods[root]['forward']))
            for i, statement in enumerate(training.body):
                if isinstance(statement, ast.Return):
                    training.body = training.body[:i+1]
                    break
            method(root, 'forward', override=training)
        inference.add(inference_reference['inference']['procedure'])
        for member in ['forward', *inference_reference.get('helper_sha256', {})]:
            evidence.append(dict(_stmt(methods[root][member]), file=files[root], scope=root+'.'+member, role='source-audited inference template'))
    elif len(aliases)==1 and _tta_wrapper(methods[root]["forward"], {'self.'+aliases[0]}):
        method(root, aliases[0])
        inference.add("fixed transformed-view averaging; optional EMA snapshot ensemble")
        evidence.append(dict(_stmt(methods[root]["forward"]), file=files[root], scope=root+".forward", role="audited inference wrapper"))
    elif _tta_wrapper(methods[root]["forward"], modules[root]):
        training = Specialize({"self.training": True}).visit(copy.deepcopy(methods[root]["forward"]))
        for i, statement in enumerate(training.body):
            if isinstance(statement, ast.Return):
                training.body = training.body[:i+1]
                break
        method(root,"forward",override=training)
        inference.add("fixed transformed-view averaging/calibration around the training architecture")
    else:
        method(root, "forward", wrapper=True)
    for name in reached - {root}:
        method(name, "forward")
    op_names = {leaf for _, _, leaf in primitive_calls}
    if not {"Conv2d", "Linear"} <= op_names:
        raise Unsupported("Not the admitted convolution/classifier composition")
    if "Flatten" in op_names:
        flattened = True
    if not flattened:
        raise Unsupported("Readout representation not established")
    grouping = set()
    for name, call, leaf in primitive_calls:
        if leaf != "Conv2d":
            continue
        group = next((k.value for k in call.keywords if k.arg == "groups"), ast.Constant(value=1))
        in_channels = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "in_channels"), None)
        g = constant(group, envs[name])
        c = constant(in_channels, envs[name])
        if g == 1:
            grouping.add("dense")
        elif (g is not UNKNOWN and c is not UNKNOWN and g == c) or ast.dump(group) == ast.dump(in_channels):
            grouping.add("depthwise")
        elif g is not UNKNOWN:
            grouping.add("grouped")
        else:
            raise Unsupported("Unresolved convolution grouping")
    pools = {leaf for leaf in op_names if "Pool" in leaf} | functional_pools
    # Presence of both max and average statistics is an explicit family basis.
    pool_basis = set()
    for p in pools:
        pool_basis.add("mean" if "Avg" in p or "avg" in p else "maximum")
    fp = seed_fingerprint(FASHION)
    fp.update(
        spatial_operator=" + ".join(sorted(grouping)) + " Conv2d",
        mixing=" + ".join(sorted(grouping)) + " Conv2d",
        channel_interaction=" + ".join(sorted(grouping)) + " convolution channel connectivity",
        normalization=" + ".join(sorted(op_names & NORMS)) or "none",
        activation=" + ".join(sorted({p for p in op_names if p.lower() in ACTS or p == 'PReLU'} | functional_acts)) or "none",
        stochasticity="; ".join((["dropout declared; rate is a setting"] if any(p.startswith("Dropout") for p in op_names) else []) + sorted(functional_dropout)) or "none",
        connectivity="; ".join((["fixed additive residual/skip connections"] if residual else []) + (["fixed channel/feature concatenation of conventional branches"] if concatenated else [])) or "sequential convolution/pooling stages and dense head",
        spatial_downsampling=" + ".join(sorted(pools)) or "no explicit pool",
        scale_representation="fixed-resolution pooling/downsampling stages" if pools else "single grid resolution",
        spatial_readout="flattened grid through an affine/pointwise dense head",
        aggregation="flattened spatial features",
        parameter_construction="free learned convolution/affine weights; standard normalization parameters",
        other="closed standard convolution/affine grammar; no additional operator",
    )
    signature = {"task": "fashion", "spatial_operator": sorted(grouping), "summary_basis": sorted(pool_basis), "readout": "flattened affine/pointwise head"}
    if root_readout_summaries:
        basis = '/'.join(label for key, label in [('mean', 'mean'), ('maximum', 'max')] if key in root_readout_summaries)
        descriptor = 'global ' + basis + ' feature descriptor'
        signature['readout'] = descriptor
        fp['spatial_readout'] = descriptor + ' through affine/pointwise classifier'
        fp['aggregation'] = descriptor
    result = {"fingerprint": fp, "family_signature": signature, "evidence": evidence,
              "inference": {"procedure": "; ".join(sorted(inference)) or "single view"}, "notes": "Every reached architecture method and constructor conforms to the reviewed standard CNN grammar."}
    if 'PReLU' in op_names:
        fp['parameter_construction'] += '; learned scalar/channel slopes for the pointwise PReLU activation'
    if storage_conversions:
        fp['other'] += '; floating-point precision or tensor storage layout conversion: ' + '; '.join(sorted(storage_conversions))
        result.setdefault('settings', {})['storage_conversions'] = sorted(storage_conversions)
    if inference_reference:
        for component, detail in inference_reference.get('fingerprint_append', {}).items():
            fp[component] = (fp.get(component, '') + '; ' + detail).strip('; ')
        fp.update(inference_reference.get('fingerprint_replace', {}))
        signature.update(inference_reference.get('family_facts', {}))
        result['notes'] += ' The full inference method and its auxiliary methods match individually source-audited templates; float coefficients are settings, while every operator, integer dimension and selected coordinate remains bound.'
        result.setdefault('settings', {})['inference_numeric_literals'] = sorted({str(n.value) for n in ast.walk(methods[root]['forward']) if isinstance(n, ast.Constant) and type(n.value) in (int, float)})
        if inference_reference.get('family_label'):
            result['family_label'] = inference_reference['family_label']
    return result


def _fixed_windows(method):
    """Accept only pure configuration/index arithmetic for fixed windows."""
    permitted = {"range", "len", "all", "min", "max", "int", "tuple", "list", "enumerate"}
    locals_ = {n.id for n in ast.walk(method) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for n in ast.walk(method):
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.Try, ast.While, ast.Yield)):
            return False
        if isinstance(n, ast.Call):
            name = dotted(n.func)
            if name in permitted:
                continue
            if isinstance(n.func, ast.Attribute) and n.func.attr == "append" and isinstance(n.func.value, ast.Name) and n.func.value.id in locals_:
                continue
            if isinstance(n.func, ast.Attribute) and n.func.attr in {"upper", "replace", "index"}:
                base = dotted(n.func.value)
                if base == "config.window_pattern" or base in locals_ or isinstance(n.func.value, ast.Constant):
                    continue
            return False
        if isinstance(n, ast.Attribute) and dotted(n).startswith("self."):
            return False
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in locals_ | permitted | {"self", "config"}:
            return False
    return True


def _nano_parts(sources):
    # nanoGPT has a fixed named entrypoint. Avoid repeatedly specializing its
    # unrelated 500-line optimizer just to compare model definitions.
    defs, files, trees = {}, {}, []
    for filename, source in sources.items():
        tree = ast.parse(source)
        trees.append(tree)
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                if node.name in defs:
                    raise Unsupported("Duplicate module-level definition")
                defs[node.name] = Specialize({}).visit(node)
                files[node.name] = filename
    classes = {name for name,node in defs.items() if isinstance(node,ast.ClassDef) and any(dotted(b).endswith(".Module") for b in node.bases)}
    roots = {"GPT"} if "GPT" in classes else set()
    if not roots:
        raise Unsupported("Unrecognized LM root")
    reached = _model_reached(defs, roots)
    pieces = {}
    evidence = []
    factory = defs.get("build_model_config")
    if factory is None:
        raise Unsupported("Unrecognized LM configuration factory")
    config_calls = [n for n in ast.walk(factory) if isinstance(n, ast.Call) and dotted(n.func) == "GPTConfig"]
    if len(config_calls) != 1:
        raise Unsupported("Multiple LM configuration alternatives")
    kwargs = {k.arg: k.value for k in config_calls[0].keywords}
    if "n_head" not in kwargs or "n_kv_head" not in kwargs:
        raise Unsupported("Head relation not specified")
    if ast.dump(kwargs["n_head"]) != ast.dump(kwargs["n_kv_head"]):
        config_env={}
        for tree in trees:assign_constants(tree.body,config_env)
        if "DEPTH" in config_env:config_env['depth']=config_env['DEPTH']
        assign_constants(factory.body,config_env)
        qh,kh=constant(kwargs['n_head'],config_env),constant(kwargs['n_kv_head'],config_env)
        if type(qh) is not int or type(kh) is not int or not (0<kh<=qh) or qh%kh:
            raise Unsupported("Head grouping cannot be resolved from selected factory")
        if qh!=kh:pieces['kv_head_relation']='multiple query heads share each key/value head'
    pieces["build_model_config"] = _key(factory)
    # The reviewed partial-RoPE helpers preserve a nonempty rotary subspace.
    # Do not infer that property for an unresolved or degenerate configuration.
    positionless = 'apply_rotary_emb' in reached and any(isinstance(a, ast.arg) and a.arg == 'positionless_head' for a in ast.walk(defs['apply_rotary_emb']))
    if positionless or 'apply_partial_rotary_emb' in reached:
        rotary_env = {}
        for tree in trees:
            assign_constants(tree.body, rotary_env)
        if 'DEPTH' in rotary_env:
            rotary_env['depth'] = rotary_env['DEPTH']
        assign_constants(factory.body, rotary_env)
        width = constant(kwargs.get('n_embd'), rotary_env)
        heads = constant(kwargs.get('n_head'), rotary_env)
        kv_heads = constant(kwargs.get('n_kv_head'), rotary_env)
        if not all(type(v) is int and v > 0 for v in (width, heads, kv_heads)) or width % heads:
            raise Unsupported('Unresolved nonempty rotary subspace')
        if positionless and min(heads, kv_heads) <= 1:
            raise Unsupported('Position-free head leaves no rotary head')
        if 'apply_partial_rotary_emb' in reached and (width // heads < 4 or (width // heads) % 4):
            raise Unsupported('Partial rotary channels do not satisfy the reviewed nonempty subspace')
    for name in sorted(reached):
        node = defs[name]
        if isinstance(node, ast.ClassDef) and name in classes:
            for m in node.body:
                if not isinstance(m, ast.FunctionDef):
                    raise Unsupported("Unrecognized LM class member")
                if m.name in {"setup_optimizer", "num_scaling_params", "estimate_flops"}:
                    continue
                if m.name == "_compute_window_sizes":
                    if not _fixed_windows(m):
                        raise Unsupported("Input-dependent window construction")
                    pieces[name + "." + m.name] = "fixed per-layer causal window schedule"
                else:
                    pieces[name + "." + m.name] = _key(m)
                evidence.append(dict(_stmt(m), file=files[name], scope=name + "." + m.name))
        elif name == "GPTConfig":
            # All fields are retained separately as numeric/configuration detail.
            # A new field is harmless only if its use is also in an audited body.
            if any(not isinstance(n, ast.AnnAssign) for n in node.body):
                raise Unsupported("Nonliteral GPT config behavior")
        elif isinstance(node, ast.FunctionDef):
            pieces[name] = _key(node)
            evidence.append(dict(_stmt(node), file=files[name], scope=name))
        else:
            raise Unsupported("Unknown LM helper")
    return pieces, evidence


def _nano(sources):
    reference = Path(__file__).with_name("ontology_vision_lm_references.json")
    if not reference.exists():
        raise Unsupported("LM source reference unavailable")
    refs = _read_json(reference)
    pieces, evidence = _nano_parts(sources)
    for ref in refs.get("nanogpt", []):
        if pieces == ref["parts"]:
            return {"fingerprint": dict(ref["fingerprint"]), "family_signature": ref["family_signature"],
                    "evidence": evidence, "notes": "Matches audited LM computation with fixed configuration-only context-window scheduling; training, widths and scheduling lengths are separate settings."}
    catalog = refs.get("nanogpt_method_catalog", {})
    facts, family_facts = {}, {}
    for scope, key in pieces.items():
        if key == "fixed per-layer causal window schedule":
            continue
        if scope == 'kv_head_relation':
            facts.setdefault('projection_relations',set()).add(key)
            continue
        entry = catalog.get(scope, {}).get(digest(key))
        if entry is None:
            raise Unsupported("Unaudited LM computation body: " + scope)
        for field, values in entry["facts"].items():
            facts.setdefault(field, set()).update(values)
        for field, values in entry["family_facts"].items():
            mechanisms = {_nano_family_fact(field, value) for value in values}
            mechanisms.discard(None)
            if mechanisms:
                family_facts.setdefault(field, set()).update(mechanisms)
    fp = seed_fingerprint(NANO)
    replace = {"state", "input_transform", "sharing", "output", "other", "feedforward", "bottleneck"}
    for field, values in facts.items():
        text = "; ".join(sorted(values))
        replace_value = field in replace or field == "kv_memory" and any("ungated" in v or "per-channel token-value gate" in v for v in values)
        fp[field] = text if replace_value else fp[field] + "; " + text
    if any('SwiGLU' in value for value in facts.get('feedforward',set())):
        fp['activation']=fp['activation'].replace('squared ReLU MLP','SiLU-gated feedforward')
        fp['block_composition']='pre-normalized attention then a multiplicatively gated SwiGLU feedforward block'
    if any('ungated token-value lookup addition' in value for value in facts.get('kv_memory',set())):
        fp['activation']=fp['activation'].replace('; sigmoid value gate','')
    if fp['output']=='affine token logits without soft cap':
        fp['activation']=fp['activation'].replace('; tanh logit soft cap','')
    family = dict(refs["nanogpt"][0]["family_signature"])
    family.update({field: sorted(values) for field, values in family_facts.items()})
    return {"fingerprint": fp, "family_signature": family, "evidence": evidence,
            "settings": {'kv_projection_sharing': pieces.get('kv_head_relation', 'separate KV projections for query heads'),
                         'fixed_attention_head_layout': '; '.join(sorted(facts.get('context_topology', set()))) or 'shared per-layer fixed window schedule',
                         'rotary_subspace': '; '.join(sorted(facts.get('position', set()))) or 'full rotary QK channels',
                         'memory_projection_parameterization': '; '.join(sorted(facts.get('bottleneck', set()))) or 'seed parameterization'},
            "notes": "Every model constructor, inference method, initializer and referenced helper matches an individually inspected computation variant. The family records the described mechanisms; widths, routine scalar/normalization choices, projection weight sharing and fixed context schedules remain settings."}


def _nano_family_fact(field, value):
    """Canonical primitive families settled against the handoff rubric.

    These exact source-audited exceptions do not cover removing every rotary
    channel, disabling a gate, input-dependent masks, new information direction,
    or new memory/history features. Their detailed source facts remain in FP.
    """
    routine = {
        'context_topology': {'parallel local-window and global-window head groups'},
        'position': {'partial rotary positions plus unrotated content channels', 'rotary positions with a position-free attention-head subspace'},
        'kv_memory': {'per-channel token-value gate'},
        'bottleneck': {'low-rank projection of prefix-mean memory'},
    }
    if value in routine.get(field, set()):
        return None
    if field == 'feedforward' and value in {'squared-ReLU MLP receives learned fixed-lag causal feature mixtures', 'squared-ReLU MLP receives a causal current/previous-token feature mixture'}:
        return 'squared-ReLU MLP receives a learned fixed-lag causal feature basis'
    if field == 'input_transform' and value == 'causal prefix-mean embedding memory passed through a low-rank nonlinear map':
        return 'causal prefix-mean embedding memory passed through a nonlinear map'
    return value


def _fashion_gate_family(value):
    if isinstance(value, dict):
        result = {key: _fashion_gate_family(item) for key, item in value.items()}
        operator = result.get('operator')
        if isinstance(operator, str):
            for activation in ('GELU', 'SiLU', 'gelu', 'silu'):
                operator = operator.replace('affine ' + activation + ' affine', 'affine pointwise affine')
            # Do not canonicalize distinct output gain functions.  A positive
            # tanh gain and a sigmoid gate have different source-level routing
            # semantics and must break a preserving family edge.
            result['operator'] = operator
        return result
    if isinstance(value, list):
        return [_fashion_gate_family(item) for item in value]
    return value


def _fashion(sources):
    result = _fashion_model(sources)
    family = result['family_signature']
    connectivity = family.get('spatial_operator')
    if isinstance(connectivity, list) and connectivity and set(connectivity) <= {'dense', 'grouped', 'depthwise'}:
        family['spatial_operator'] = 'learned local Conv2d'
        result.setdefault('settings', {})['convolution_connectivity'] = ', '.join(connectivity)
        result['notes'] += ' Dense, grouped and depthwise local convolution retain the same primitive family; their coefficient connectivity remains explicit in the fingerprint.'
    if family.get('spatial_context') == 'parallel local and dilated branches':
        # This exact descriptor comes from fully reviewed free Conv2d branches.
        # Their fixed concatenation and dilation do not add a representation
        # primitive. Do not erase attention, fixed filters or input-dependent
        # branch routing under this narrow parameter-connectivity equivalence.
        result.setdefault('settings', {})['spatial_connectivity'] = family.pop('spatial_context')
        result['notes'] += ' Parallel fixed local/dilated learned convolution branches are ordinary convolution connectivity; exact geometry remains in the fingerprint.'
    result['family_signature'] = _fashion_gate_family(family)
    return result


def profile(sources, campaign_key):
    """Return a full component profile only when all model code is admitted."""
    if not sources:
        return None
    try:
        return _fashion(sources) if campaign_key == FASHION else _nano(sources) if campaign_key == NANO else None
    except (Unsupported, ValueError, SyntaxError, TypeError, KeyError):
        return None


def review_pair(before_sources, after_sources, campaign_key):
    if campaign_key == FASHION:
        staged = _staged_fashion_candidate_pair(before_sources, after_sources)
        if staged is not None:
            return staged
    before = profile(before_sources, campaign_key) if before_sources else None
    after = profile(after_sources, campaign_key)
    if after is None:
        return None
    parent_pending = before_sources is not None and before is None
    changed = sorted(k for k in set(after["family_signature"]) | (set(before["family_signature"]) if before else set()) if before and before["family_signature"].get(k) != after["family_signature"].get(k))
    return {"classification": "uncertain" if parent_pending else "changing" if changed else "preserving", "fingerprint": after["fingerprint"],
            "fingerprint_complete": True, "family_signature": after["family_signature"], "changed_components": changed,
            "notes": after["notes"] + (" The complete candidate fingerprint is resolved; its recorded parent still needs semantic review before a transition label can be assigned." if parent_pending else ""), "evidence": after["evidence"], "inference": after.get("inference", {}),
            "family_label": after.get("family_label"),
            "training": after.get("training", {}), "settings": after.get("settings", {}),
            "review_method": VERSION, "source_sha256": digest(after_sources),
            "parent_source_sha256": digest(before_sources) if before_sources else None}





