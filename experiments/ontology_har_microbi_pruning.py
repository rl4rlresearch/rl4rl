"""Exact-source proof for one HAR MicroBiConvLSTM inference-pruning family.

The proof is intentionally narrower than a class-name match.  Both programs
must reduce to the reviewed whole-program AST after replacing only three
source-visible settings of an already-present ranked feature-pruning path:
the ``torch.topk(..., k=...)`` count, its corresponding selected rank, and
the dependent input width of one one-logit affine head.  Every other literal,
operator, branch, selector, module, and training hook remains hash-locked.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json

from experiments.review_ontology_sources import CORE, TASK_KEYS, dotted


_TASK = "har"


def _is_microbi_pruning_class(node: ast.ClassDef) -> bool:
    return node.name == "MicroBiConvLSTM" and any(
        isinstance(item, ast.FunctionDef) and item.name == "forward"
        for item in node.body
    )


class _ReviewedPruningSlots(ast.NodeTransformer):
    """Replace only the three independently checked coordinate-pruning slots."""

    def __init__(self) -> None:
        self.topk_counts = 0
        self.rank_selectors = 0
        self.affine_widths = 0

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if dotted(node.func) == "torch.topk":
            first = node.args[0] if node.args else None
            risk_name = (
                first.id if isinstance(first, ast.Name) else
                first.func.value.id if isinstance(first, ast.Call)
                and isinstance(first.func, ast.Attribute)
                and first.func.attr == "index_select"
                and isinstance(first.func.value, ast.Name) else None
            )
            selected_risk = (
                risk_name in {"initialSecondaryConnectionRisk", "secondaryConnectionRisk"}
            )
            for keyword in node.keywords:
                if selected_risk and keyword.arg == "k" and isinstance(keyword.value, ast.Constant):
                    if type(keyword.value.value) is not int or keyword.value.value < 1:
                        raise ValueError("reviewed top-k count must be a positive integer")
                    keyword.value = ast.Constant(value="reviewed ranked-pruning count")
                    self.topk_counts += 1
        return node

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        if (
            len(node.targets) == 1
            and dotted(node.targets[0]) == "self.inferenceSecondaryReducedClassifier"
            and isinstance(node.value, ast.Call)
            and dotted(node.value.func) == "nn.Linear"
            and node.value.args
        ):
            value = node.value.args[0]
            if not (
                isinstance(value, ast.BinOp)
                and isinstance(value.op, ast.Sub)
                and dotted(value.left) == "self.inferenceFeatures"
                and isinstance(value.right, ast.Constant)
                and type(value.right.value) is int
                and value.right.value > 0
            ):
                raise ValueError("reviewed affine-pruning width is malformed")
            value.right = ast.Constant(value="reviewed ranked-pruning count")
            self.affine_widths += 1
        return node

    def visit_Subscript(self, node: ast.Subscript):
        node = self.generic_visit(node)
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "values"
            and isinstance(node.value.value, ast.Call)
            and dotted(node.value.value.func) == "torch.topk"
            and node.value.value.args
            and isinstance(node.value.value.args[0], ast.Name)
            and node.value.value.args[0].id in {"initialSecondaryConnectionRisk", "secondaryConnectionRisk"}
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) == 2
            and isinstance(node.slice.elts[1], ast.Constant)
        ):
            rank = node.slice.elts[1].value
            if type(rank) is not int or rank < 0:
                raise ValueError("reviewed top-k selector must be a nonnegative integer")
            node.slice.elts[1] = ast.Constant(value="reviewed ranked-pruning selector")
            self.rank_selectors += 1
        return node


def _template(sources: dict[str, str]) -> tuple[str, dict[str, int]] | None:
    """Return the reviewed skeleton only for the bounded MicroBi source form."""
    if set(sources) != {"train.py"}:
        return None
    try:
        tree = ast.parse(sources["train.py"])
    except SyntaxError:
        return None
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    reviewed = [node for node in classes if _is_microbi_pruning_class(node)]
    builds = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_model"]
    if len(reviewed) != 1 or len(builds) != 1:
        return None
    factory_calls = [
        node for node in ast.walk(builds[0])
        if isinstance(node, ast.Call) and dotted(node.func) == "MicroBiConvLSTM"
    ]
    if len(factory_calls) != 1:
        return None
    normalizer = _ReviewedPruningSlots()
    try:
        normalized = normalizer.visit(copy.deepcopy(tree))
    except ValueError:
        return None
    # This source form has paired initial/runtime rank selection plus their
    # matching selected-coordinate lists, and one dependent affine width.
    if (normalizer.topk_counts, normalizer.rank_selectors, normalizer.affine_widths) != (4, 2, 1):
        return None
    payload = ast.dump(ast.fix_missing_locations(normalized), include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), {
        "topk_counts": normalizer.topk_counts,
        "rank_selectors": normalizer.rank_selectors,
        "affine_widths": normalizer.affine_widths,
    }


def _fingerprint() -> dict[str, str]:
    """Complete description of the reviewed skeleton; settings are not inferred."""
    fp = dict.fromkeys(CORE + TASK_KEYS[_TASK].split())
    fp.update(
        input_units="fixed nine-channel continuous human-activity sensor windows",
        input_transform="source-defined raw-channel summary features and the selected temporal convolution path",
        embedding="none: continuous sensor frames enter convolutional operators directly",
        position="implicit temporal order through source-fixed one-dimensional convolutions and recurrent scan",
        mixing="source-fixed temporal convolutional stem and recurrent feature mixing",
        routing="fixed source-defined convolution, pooling, recurrence, summary, and ranked inference-feature-pruning paths",
        state="the source-defined recurrent hidden state; no persistent state between independent windows",
        feedforward="source-defined affine classifier and inference-only relative-logit heads",
        parameter_construction="ordinary learned convolution, recurrent, normalization, and affine parameters; inference heads are derived by the source-defined ranked coordinate procedure",
        sharing="each declared module is reused at its source call sites; no cross-module alias is admitted by this exact-source proof",
        normalization="the source-defined normalization modules in the reviewed whole-program skeleton",
        connectivity="temporal stem, pooling, recurrence, raw summaries, classifier, and fixed ranked coordinate-pruning readout as specified by the reviewed AST",
        aggregation="source-defined recurrent/readout summaries followed by the existing ranked coordinate-pruning readout",
        output="six activity-logit coordinates through the existing source-defined relative-logit inference head",
        symmetry="no input permutation or rotation canonicalization is introduced by the reviewed skeleton",
        conditional_compute="fixed source control flow; ranked feature selection is an existing inference-head construction, not input-dependent routing",
        activation="the source-defined nonlinearities in the reviewed temporal and recurrent path",
        stochasticity="source-defined classifier dropout; no new inference-time random branch",
        bottleneck="the source-defined temporal/recurrent widths and already-present ranked inference feature selection",
        iteration="source-defined finite temporal convolution stages and recurrent scan",
        other="Complete exact-source proof: outside the reviewed ranked-pruning slots, the entire parsed train.py AST is identical after docstrings are removed.",
        sensor_fusion="source-defined convolutional and recurrent mixing of the nine ordered sensor channels",
        temporal_operator="the unchanged source-defined Conv1d, pooling, and recurrent operators",
        directionality="the unchanged source-defined recurrent directionality",
        state_update="the unchanged source-defined recurrent update",
        frequency_representation="no newly introduced spectral transform; the reviewed program retains its time-domain operators",
        temporal_readout="the unchanged source-defined summaries and relative-logit coordinate-pruning head",
    )
    return fp


def review_pair(before_sources: dict[str, str], after_sources: dict[str, str]):
    """Prove a preserving transition for the reviewed finite source family."""
    before = _template(before_sources)
    after = _template(after_sources)
    if before is None or after is None or before[0] != after[0]:
        return None
    return {
        "classification": "preserving",
        "fingerprint": _fingerprint(),
        "fingerprint_complete": True,
        "residual_components": [],
        "changed_components": [],
        "notes": "Exact source proof: both recorded programs match the reviewed MicroBiConvLSTM ranked inference-feature-pruning skeleton. Only the existing top-k pruning count, its matching rank selector, and the dependent one-logit affine input width differ.",
        "evidence": [{
            "file": "train.py",
            "line": 1,
            "code": "Whole-program AST matches after replacing only four reviewed torch.topk counts, two matching .values rank selectors, and one dependent inferenceFeatures-minus-count affine width.",
            "kind": "complete parent-to-candidate exact-source preserving proof",
        }],
        "settings": {"reviewed_slots": after[1]},
        "family_label": "MicroBiConvLSTM / ranked relative-logit inference-feature pruning",
        "family_signature": {
            "temporal_path": "reviewed fixed MicroBiConvLSTM AST",
            "readout": "existing ranked inference-feature-pruning relative-logit head",
            "transition": "rank count and dependent coordinate width only",
        },
        "reviewer": "har-microbi-ranked-pruning-exact-source-v1",
    }
