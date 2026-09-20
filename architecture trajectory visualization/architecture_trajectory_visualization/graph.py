"""Conservative, provider-free architecture descriptions from saved source or IR.

Candidate code is parsed, never imported, evaluated, compiled, or executed. Python
descriptors describe declared modules and a partial set of source-proven call
dependencies, not a traced tensor graph. ``nn.Sequential`` order is meaningful;
``ModuleList`` order and the order of unrelated declarations are not data flow.

Public API: ``extract_architecture(sources, *, source_id=None, source_hash=None,
ir=None)`` and ``diff_architectures(before, after)``. All outputs are JSON values.
Node ids ignore Python identifiers and source line numbers so ordinary renames
are stable. The diff additionally matches structural signatures before ids.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "1.0"
_MODULE_KINDS = {
    "Embedding": "embedding",
    "EmbeddingBag": "embedding",
    "MultiheadAttention": "attention",
    "TransformerEncoder": "attention",
    "TransformerDecoder": "attention",
    "TransformerEncoderLayer": "attention",
    "TransformerDecoderLayer": "attention",
    "Linear": "feed_forward",
    "Bilinear": "feed_forward",
    "LayerNorm": "normalization",
    "RMSNorm": "normalization",
    "GroupNorm": "normalization",
    "BatchNorm1d": "normalization",
    "BatchNorm2d": "normalization",
    "BatchNorm3d": "normalization",
    "InstanceNorm1d": "normalization",
    "InstanceNorm2d": "normalization",
    "Conv1d": "convolution",
    "Conv2d": "convolution",
    "Conv3d": "convolution",
    "ConvTranspose1d": "convolution",
    "ConvTranspose2d": "convolution",
    "MaxPool1d": "pooling",
    "MaxPool2d": "pooling",
    "MaxPool3d": "pooling",
    "AvgPool1d": "pooling",
    "AvgPool2d": "pooling",
    "AvgPool3d": "pooling",
    "AdaptiveAvgPool1d": "pooling",
    "AdaptiveAvgPool2d": "pooling",
    "AdaptiveMaxPool1d": "pooling",
    "AdaptiveMaxPool2d": "pooling",
    "RNN": "recurrent",
    "GRU": "recurrent",
    "LSTM": "recurrent",
    "RNNCell": "recurrent",
    "GRUCell": "recurrent",
    "LSTMCell": "recurrent",
    "Sequential": "container",
    "ModuleList": "container",
    "ModuleDict": "container",
    "Parameter": "parameter",
    "ParameterList": "container",
    "ParameterDict": "container",
    "ReLU": "activation",
    "GELU": "activation",
    "SiLU": "activation",
    "Tanh": "activation",
    "Sigmoid": "activation",
    "Softmax": "routing",
    "LogSoftmax": "routing",
    "Dropout": "stochasticity",
    "Dropout2d": "stochasticity",
    "Flatten": "reshape",
    "Identity": "custom",
}
_ARG_NAMES = {
    "Linear": ["in_features", "out_features", "bias"],
    "Embedding": ["num_embeddings", "embedding_dim", "padding_idx"],
    "MultiheadAttention": ["embed_dim", "num_heads", "dropout"],
    "LayerNorm": ["normalized_shape", "eps", "elementwise_affine"],
    "RMSNorm": ["normalized_shape", "eps", "elementwise_affine"],
    "Conv1d": [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
    ],
    "Conv2d": [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
    ],
    "GRU": [
        "input_size",
        "hidden_size",
        "num_layers",
        "bias",
        "batch_first",
        "dropout",
        "bidirectional",
    ],
    "LSTM": [
        "input_size",
        "hidden_size",
        "num_layers",
        "bias",
        "batch_first",
        "dropout",
        "bidirectional",
    ],
    "RNN": [
        "input_size",
        "hidden_size",
        "num_layers",
        "nonlinearity",
        "bias",
        "batch_first",
    ],
    "GRUCell": ["input_size", "hidden_size", "bias"],
    "LSTMCell": ["input_size", "hidden_size", "bias"],
}
_FUNCTION_KINDS = {
    "relu": "activation",
    "gelu": "activation",
    "silu": "activation",
    "tanh": "activation",
    "sigmoid": "activation",
    "softmax": "routing",
    "log_softmax": "routing",
    "linear": "feed_forward",
    "layer_norm": "normalization",
    "rms_norm": "normalization",
    "conv1d": "convolution",
    "conv2d": "convolution",
    "scaled_dot_product_attention": "attention",
    "cat": "fusion",
    "stack": "fusion",
    "flatten": "reshape",
    "mean": "pooling",
    "sum": "aggregation",
    "matmul": "mixing",
    "einsum": "mixing",
}


def _json(value: Any) -> str:
    return json.dumps(
        _safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"non_finite": str(value), "value": None}
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and abs(value) > 2**53 - 1
    ):
        return {"integer": str(value)}
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name(node.value) + "." + node.attr
    if isinstance(node, ast.Subscript):
        return _name(node.value) + "[" + _expression(node.slice) + "]"
    return ""


def _expression(node: ast.AST | None) -> str:
    try:
        return ast.unparse(node)[:400] if node is not None else ""
    except (RecursionError, ValueError):
        return "<expression>"


def _value(node: ast.AST, env: Mapping[str, Any]) -> Any:
    """Read literals/defaults only. Unresolved expressions remain explicit strings."""
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, int, float, bool, type(None))
    ):
        if isinstance(node.value, int) and abs(node.value) > 2**53 - 1:
            return {"integer": str(node.value)}
        return _safe(node.value)
    if _name(node) in env:
        return env[_name(node)]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_value(item, env) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            str(_value(key, env)): _value(value, env)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _value(node.operand, env)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return -value if isinstance(node.op, ast.USub) else value
    return {"expression": _expression(node), "resolved": False}


def _evidence(path: str, node: ast.AST, method: str = "source") -> dict[str, Any]:
    return {
        "file": path,
        "line": getattr(node, "lineno", None),
        "end_line": getattr(node, "end_lineno", None),
        "expression": _expression(node),
        "method": method,
    }


def _kind(operation: str) -> str:
    short = operation.rsplit(".", 1)[-1]
    if short in _MODULE_KINDS:
        return _MODULE_KINDS[short]
    # These are labeled lexical hints, never evidence of a concrete internal graph.
    lower = short.lower()
    if "attention" in lower:
        return "attention"
    if any(token in lower for token in ("mlp", "feedforward", "feed_forward")):
        return "feed_forward"
    return "custom"


class _Builder:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.groups: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.counters: Counter[tuple[str, str]] = Counter()
        self.edge_keys: set[tuple[str, str, str, str]] = set()
        self.classes: set[str] = set()

    def node(
        self,
        *,
        group: str,
        kind: str,
        label: str,
        operation: str,
        evidence: dict[str, Any],
        config: dict[str, Any] | None = None,
        role: str = "module",
    ) -> dict[str, Any]:
        index = self.counters[(group, kind)]
        self.counters[(group, kind)] += 1
        node = {
            "id": f"{group}/{kind}/{index}",
            "kind": kind,
            "label": label,
            "operation": operation,
            "group_id": group,
            "config": config or {},
            "evidence": [evidence],
            "confidence": "source_declared",
            "role": role,
            "config_context": "source expressions and definition defaults; "
            "not resolved instance arguments",
        }
        self.nodes.append(node)
        return node

    def edge(
        self,
        source: str,
        target: str,
        relation: str,
        evidence: dict[str, Any],
        attributes: dict[str, Any] | None = None,
    ) -> None:
        key = (source, target, relation, _json(attributes or {}))
        if source == target or key in self.edge_keys:
            return
        self.edge_keys.add(key)
        self.edges.append(
            {
                "id": "edge-" + _digest(key)[:16],
                "source": source,
                "target": target,
                "type": relation,
                "evidence": [evidence],
                "confidence": "source_declared",
                "attributes": attributes or {},
            }
        )

    def module(
        self, call: ast.Call, label: str, group: str, path: str, env: Mapping[str, Any]
    ) -> dict[str, Any]:
        operation = _name(call.func)
        short = operation.rsplit(".", 1)[-1]
        kind = _kind(operation)
        names = _ARG_NAMES.get(short, [])
        config = (
            {
                names[i] if i < len(names) else f"arg_{i}": _value(arg, env)
                for i, arg in enumerate(call.args)
            }
            if kind != "container"
            else {}
        )
        config.update(
            {kw.arg or "**kwargs": _value(kw.value, env) for kw in call.keywords}
        )
        node = self.node(
            group=group,
            kind=kind,
            label=label,
            operation=operation,
            evidence=_evidence(path, call),
            config=config,
        )
        if short in {"GRU", "LSTM", "RNN", "GRUCell", "LSTMCell", "RNNCell"}:
            node["recurrence"] = {
                "kind": "declared_recurrent_module",
                "execution_steps": None,
            }
        if kind == "container":
            child_group = node["id"] + "/members"
            self.groups.append(
                {
                    "id": child_group,
                    "label": label + " members",
                    "parent_id": group,
                    "container_node_id": node["id"],
                    "order_is_execution": short == "Sequential",
                }
            )
            children: list[dict[str, Any]] = []
            values = list(call.args)
            if len(values) == 1 and isinstance(values[0], (ast.List, ast.Tuple)):
                values = list(values[0].elts)
            if len(values) == 1 and isinstance(values[0], ast.ListComp):
                comp = values[0]
                values = [comp.elt]
                node["repeat"] = {
                    "expression": _expression(comp.generators[0].iter),
                    "count": None,
                    "semantics": "declaration_template_not_execution",
                }
                iterator = comp.generators[0].iter
                if (
                    isinstance(iterator, ast.Call)
                    and _name(iterator.func) == "range"
                    and len(iterator.args) == 1
                ):
                    count = _value(iterator.args[0], env)
                    if (
                        isinstance(count, int)
                        and not isinstance(count, bool)
                        and count >= 0
                    ):
                        node["repeat"]["count"] = count
                self.warnings.append(
                    f"{label}: comprehension shown as a declaration template; "
                    "members are not expanded."
                )
            for index, value in enumerate(values):
                if isinstance(value, ast.Call):
                    child = self.module(
                        value, f"{label}[{index}]", child_group, path, env
                    )
                    children.append(child)
                    self.edge(
                        node["id"], child["id"], "containment", _evidence(path, value)
                    )
            if (
                short == "Sequential"
                and not node.get("repeat")
                and len(children) == len(values)
            ):
                for left, right in zip(children, children[1:], strict=False):
                    self.edge(
                        left["id"],
                        right["id"],
                        "data_flow",
                        _evidence(path, call, "sequential_declaration"),
                    )
        return node

    def parse_class(
        self, cls: ast.ClassDef, index: int, path: str, constants: Mapping[str, Any]
    ) -> None:
        group = f"class-{index}"
        self.groups.append(
            {
                "id": group,
                "label": cls.name,
                "parent_id": None,
                "source_file": path,
                "evidence": [_evidence(path, cls)],
                "role": "class_definition",
            }
        )
        methods = {
            node.name: node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        init = methods.get("__init__")
        env = dict(constants)
        modules: dict[str, dict[str, Any]] = {}
        ties: list[tuple[str, str, ast.AST]] = []
        if init:
            for arg, default in zip(
                init.args.args[-len(init.args.defaults) :]
                if init.args.defaults
                else [],
                init.args.defaults,
                strict=True,
            ):
                env[arg.arg] = _value(default, env)
            for statement in ast.walk(init):
                if (
                    not isinstance(statement, (ast.Assign, ast.AnnAssign))
                    or statement.value is None
                ):
                    continue
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                value = statement.value
                for target in targets:
                    name = _name(target)
                    if not name:
                        continue
                    if (
                        name.startswith("self.")
                        and name.count(".") > 1
                        and _name(value).startswith("self.")
                    ):
                        ties.append((name, _name(value), statement))
                    elif name.startswith("self.") and isinstance(value, ast.Call):
                        operation = _name(value.func)
                        short = operation.rsplit(".", 1)[-1]
                        if (
                            short in _MODULE_KINDS
                            or short in self.classes
                            or operation.startswith(("nn.", "torch.nn."))
                        ):
                            modules[name] = self.module(
                                value, name.removeprefix("self."), group, path, env
                            )
                        else:
                            env[name] = _value(value, env)
                    elif name.startswith("self.") and _name(value) in modules:
                        modules[name] = modules[_name(value)]
                        modules[name].setdefault("aliases", []).append(
                            name.removeprefix("self.")
                        )
                    else:
                        env[name] = _value(value, env)
        for target, source, statement in ties:
            target_owner = target.rsplit(".", 1)[0]
            source_owner = source.rsplit(".", 1)[0]
            if target_owner in modules and source_owner in modules:
                left, right = modules[target_owner], modules[source_owner]
                self.edge(
                    left["id"],
                    right["id"],
                    "shared_parameters",
                    _evidence(path, statement),
                    {
                        "source_parameter": target.rsplit(".", 1)[-1],
                        "target_parameter": source.rsplit(".", 1)[-1],
                    },
                )
                for node, own, other in [
                    (left, target, source),
                    (right, source, target),
                ]:
                    node.setdefault("sharing", []).append(
                        {
                            "attribute": own.rsplit(".", 1)[-1],
                            "with": other,
                            "evidence": _evidence(path, statement),
                        }
                    )
            else:
                self.warnings.append(
                    f"{path}:{statement.lineno}: parameter alias recorded but "
                    "owner modules could not be resolved."
                )
        forward = methods.get("forward")
        if forward:
            self.forward(forward, modules, group, path, env)
        elif "recurrent_step" in methods:
            next(item for item in self.groups if item["id"] == group)[
                "interface_method"
            ] = "recurrent_step"
            self.forward(methods["recurrent_step"], modules, group, path, env)
            self.warnings.append(
                f"{cls.name}: recurrent_step interface inspected; external execution "
                "loop and classify method are not a single traced graph."
            )
        elif modules:
            self.warnings.append(f"{cls.name}: no forward method; declarations only.")

    def forward(
        self,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        modules: Mapping[str, dict[str, Any]],
        group: str,
        path: str,
        env: Mapping[str, Any],
    ) -> None:
        symbols: dict[str, set[str]] = {}
        for arg in method.args.args:
            if arg.arg == "self":
                continue
            node = self.node(
                group=group,
                kind="input",
                label=arg.arg,
                operation="argument",
                evidence=_evidence(path, arg),
                role="interface",
            )
            symbols[arg.arg] = {node["id"]}
        call_counts: Counter[str] = Counter()
        for candidate in ast.walk(method):
            if isinstance(candidate, ast.Call) and _name(candidate.func) in modules:
                node = modules[_name(candidate.func)]
                call_counts[node["id"]] += 1
                node.setdefault("call_sites", []).append(_evidence(path, candidate))

        def expression(expr: ast.AST) -> set[str]:
            if isinstance(expr, ast.Name):
                return symbols.get(expr.id, set())
            if (
                _name(expr) in modules
                and modules[_name(expr)].get("kind") == "parameter"
            ):
                return {modules[_name(expr)]["id"]}
            if isinstance(expr, (ast.Tuple, ast.List)):
                return set().union(*(expression(item) for item in expr.elts))
            if isinstance(expr, ast.Call):
                name = _name(expr.func)
                module = modules.get(name)
                if module:
                    deps = set().union(
                        *(expression(arg) for arg in expr.args),
                        *(expression(kw.value) for kw in expr.keywords),
                    )
                    for dep in deps:
                        self.edge(
                            dep,
                            module["id"],
                            "data_flow",
                            _evidence(path, expr, "direct_forward_call"),
                        )
                    return {module["id"]}
                short = name.rsplit(".", 1)[-1]
                # Represent explicit operations rather than silently bypassing them.
                if short in _FUNCTION_KINDS or (
                    isinstance(expr.func, ast.Attribute)
                    and short
                    in {
                        "view",
                        "reshape",
                        "transpose",
                        "permute",
                        "contiguous",
                        "unsqueeze",
                        "squeeze",
                    }
                ):
                    deps = set().union(*(expression(arg) for arg in expr.args))
                    if isinstance(expr.func, ast.Attribute):
                        deps |= expression(expr.func.value)
                    node = self.node(
                        group=group,
                        kind=_FUNCTION_KINDS.get(short, "reshape"),
                        label=short,
                        operation=short,
                        evidence=_evidence(path, expr),
                        role="operation",
                        config={
                            kw.arg or "**kwargs": _value(kw.value, env)
                            for kw in expr.keywords
                        },
                    )
                    for dep in deps:
                        self.edge(
                            dep,
                            node["id"],
                            "data_flow",
                            _evidence(path, expr, "direct_forward_expression"),
                        )
                    return {node["id"]}
                self.warnings.append(
                    f"{path}:{expr.lineno}: call {_expression(expr.func)} "
                    "is not resolved; flow stops here."
                )
                return set()
            if isinstance(expr, ast.BinOp):
                deps = expression(expr.left) | expression(expr.right)
                if not deps:
                    return set()
                operation = type(expr.op).__name__.lower()
                node = self.node(
                    group=group,
                    kind="merge" if isinstance(expr.op, ast.Add) else "custom",
                    label=operation,
                    operation=operation,
                    evidence=_evidence(path, expr),
                    role="operation",
                )
                for dep in deps:
                    self.edge(
                        dep,
                        node["id"],
                        "data_flow",
                        _evidence(path, expr, "direct_forward_expression"),
                    )
                return {node["id"]}
            if isinstance(expr, ast.Subscript):
                deps = expression(expr.value)
                if not deps:
                    return set()
                node = self.node(
                    group=group,
                    kind="reshape",
                    label="index",
                    operation="getitem",
                    evidence=_evidence(path, expr),
                    role="operation",
                    config={"index": _expression(expr.slice)},
                )
                for dep in deps:
                    self.edge(dep, node["id"], "data_flow", _evidence(path, expr))
                return {node["id"]}
            return set()

        for statement in method.body:
            if (
                isinstance(statement, (ast.Assign, ast.AnnAssign))
                and statement.value is not None
            ):
                values = expression(statement.value)
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        symbols[target.id] = values
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        # Tuple-result element semantics require tracing.
                        for item in target.elts:
                            if isinstance(item, ast.Name):
                                symbols[item.id] = set()
                        self.warnings.append(
                            f"{path}:{statement.lineno}: unpacked result "
                            "not resolved into tensor outputs."
                        )
            elif isinstance(statement, ast.Return) and statement.value is not None:
                returned = expression(statement.value)
                for node in self.nodes:
                    if node["id"] in returned:
                        node["is_output"] = True
            elif isinstance(statement, ast.Expr):
                expression(statement.value)
            elif isinstance(
                statement,
                (
                    ast.For,
                    ast.While,
                    ast.If,
                    ast.Try,
                    ast.With,
                    ast.Match,
                    ast.AugAssign,
                ),
            ):
                # Prevent later calls leaping across omitted control flow.
                for inner in ast.walk(statement):
                    if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store):
                        symbols[inner.id] = set()
                self.warnings.append(
                    f"{path}:{statement.lineno}: {type(statement).__name__} "
                    "control/update structure is not expanded; "
                    "downstream flow may be incomplete."
                )
                if isinstance(statement, (ast.For, ast.While)):
                    for inner in ast.walk(statement):
                        if isinstance(inner, ast.Call) and _name(inner.func) in modules:
                            modules[_name(inner.func)].setdefault(
                                "execution", []
                            ).append(
                                {
                                    "kind": "call_inside_loop",
                                    "iterations": None,
                                    "evidence": _evidence(path, statement),
                                }
                            )
        repeated = {
            identifier for identifier, count in call_counts.items() if count > 1
        }
        if repeated:
            self.edges = [
                edge
                for edge in self.edges
                if edge["type"] != "data_flow"
                or (edge["source"] not in repeated and edge["target"] not in repeated)
            ]
            self.warnings.append(
                "Repeated calls to one declared module are recorded as call sites; "
                "its execution flow is omitted rather than collapsed into one call."
            )


def _from_ir(ir: Mapping[str, Any]) -> dict[str, Any] | None:
    graph = ir.get("graph", ir)
    if (
        not isinstance(graph, Mapping)
        or not isinstance(graph.get("nodes"), list)
        or not graph["nodes"]
        or not isinstance(graph.get("groups", []), list)
    ):
        return None
    nodes = []
    warnings = []
    seen: set[str] = set()
    for raw in graph["nodes"]:
        if not isinstance(raw, Mapping) or not isinstance(
            raw.get("id", raw.get("node_id")), (str, int)
        ):
            return None
        identifier = str(raw.get("id", raw.get("node_id")))
        if not identifier or identifier in seen or isinstance(raw.get("id"), bool):
            return None
        seen.add(identifier)
        operation = str(
            raw.get(
                "op", raw.get("operation", raw.get("kind", raw.get("type", "custom")))
            )
        )
        kind = raw.get("kind", _kind(operation))
        config = raw.get("config", raw.get("attributes", raw.get("params", {})))
        if not isinstance(kind, str) or not kind or not isinstance(config, Mapping):
            return None
        node = {
            "id": identifier,
            "kind": "embedding" if kind == "token_embedding" else kind,
            "label": str(raw.get("label", raw.get("name", identifier))),
            "operation": operation,
            "group_id": raw.get("group_id"),
            "config": dict(config),
            "evidence": [{"method": "serialized_ir", "node_id": identifier}],
            "confidence": "recorded_ir",
            "role": "operation",
        }
        for field in ("shape", "sharing", "recurrence"):
            if field in raw:
                node[field] = raw[field]
        if "output_shape" in raw:
            node["shape"] = {
                "inputs": raw.get("input_shapes", []),
                "output": raw["output_shape"],
            }
        if identifier == graph.get("output_node_id"):
            node["is_output"] = True
        nodes.append(node)
    raw_edges = graph.get("edges", [])
    if not isinstance(raw_edges, list):
        return None
    raw_edges = list(raw_edges)
    if not raw_edges:
        for raw in graph["nodes"]:
            for source in (
                raw.get("inputs", []) if isinstance(raw.get("inputs", []), list) else []
            ):
                if isinstance(source, (str, int)):
                    raw_edges.append(
                        {
                            "source": str(source),
                            "target": str(raw.get("id", raw.get("node_id"))),
                            "type": "data_flow",
                        }
                    )
    edges = []
    for raw in raw_edges:
        if not isinstance(raw, Mapping):
            return None
        source, target = str(raw.get("source", "")), str(raw.get("target", ""))
        if source not in seen or target not in seen:
            warnings.append(
                f"IR edge {source} → {target} references an unavailable node."
            )
            continue
        relation = str(raw.get("type", raw.get("kind", "data_flow")))
        relation = "data_flow" if relation == "data" else relation
        edges.append(
            {
                "id": "edge-"
                + _digest([source, target, relation, raw.get("target_port")])[:16],
                "source": source,
                "target": target,
                "type": relation,
                "evidence": [{"method": "serialized_ir"}],
                "confidence": "recorded_ir",
            }
        )
        if "target_port" in raw:
            edges[-1]["target_port"] = raw["target_port"]
    for node in nodes:
        config = node.get("config")
        tied = config.get("tie_embedding") if isinstance(config, Mapping) else None
        if isinstance(tied, str) and tied in seen:
            edges.append(
                {
                    "id": "edge-"
                    + _digest([node["id"], tied, "shared_parameters"])[:16],
                    "source": node["id"],
                    "target": tied,
                    "type": "shared_parameters",
                    "evidence": [
                        {"method": "serialized_ir", "attribute": "tie_embedding"}
                    ],
                    "confidence": "recorded_ir",
                }
            )
    return {
        "nodes": nodes,
        "edges": edges,
        "groups": graph.get("groups", []),
        "representation": "computation_graph",
        "extraction": {
            "method": "serialized_ir",
            "completeness": "partial" if warnings else "recorded",
            "warnings": warnings,
            "note": "Serialized graph, not re-executed or independently traced.",
        },
    }


def extract_architecture(
    sources: Mapping[str, str],
    *,
    source_id: str | None = None,
    source_hash: str | None = None,
    ir: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON descriptor without evaluating any candidate-controlled code."""
    clean_sources = {
        str(path): text for path, text in sources.items() if isinstance(text, str)
    }
    digest = source_hash or _digest({"sources": clean_sources, "ir": ir})
    result = _from_ir(ir) if isinstance(ir, Mapping) else None
    if result is None:
        builder = _Builder()
        parsed = []
        outside_calls: set[str] = set()
        factory_calls: set[str] = set()
        for path, source in sorted(clean_sources.items()):
            if not path.endswith(".py"):
                continue
            try:
                tree = ast.parse(source, filename=path)
            except (SyntaxError, ValueError, RecursionError) as exc:
                builder.warnings.append(
                    f"{path}: parse unavailable ({type(exc).__name__})."
                )
                continue
            constants: dict[str, Any] = {}
            for statement in tree.body:
                if (
                    isinstance(statement, (ast.Assign, ast.AnnAssign))
                    and statement.value is not None
                ):
                    targets = (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else [statement.target]
                    )
                    for target in targets:
                        if isinstance(target, ast.Name):
                            constants[target.id] = _value(statement.value, constants)
            classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
            builder.classes.update(cls.name for cls in classes)
            parsed.extend((path, cls, constants) for cls in classes)
            for statement in tree.body:
                if isinstance(statement, ast.ClassDef):
                    continue
                calls = {
                    _name(item.func).rsplit(".", 1)[-1]
                    for item in ast.walk(statement)
                    if isinstance(item, ast.Call)
                }
                outside_calls.update(calls)
                if isinstance(statement, ast.FunctionDef) and statement.name in {
                    "build_model",
                    "create_model",
                    "make_model",
                    "get_model",
                }:
                    factory_calls.update(calls)
        # Ignore pure configuration/data classes; explicitly declared NN classes and
        # classes with a forward method are architecture evidence.
        selected = [
            (path, cls, constants)
            for path, cls, constants in parsed
            if any(_name(base).rsplit(".", 1)[-1] == "Module" for base in cls.bases)
            or any(
                isinstance(item, ast.FunctionDef) and item.name == "forward"
                for item in cls.body
            )
        ]
        class_names = {cls.name for _, cls, _ in selected}
        roots = (factory_calls & class_names) or (outside_calls & class_names)
        if roots:
            references = {
                cls.name: {
                    _name(item.func).rsplit(".", 1)[-1]
                    for item in ast.walk(cls)
                    if isinstance(item, ast.Call)
                }
                | {_name(base).rsplit(".", 1)[-1] for base in cls.bases}
                for _, cls, _ in selected
            }
            reachable = set(roots)
            pending = list(roots)
            while pending:
                for name in references.get(pending.pop(), set()) & class_names:
                    if name not in reachable:
                        reachable.add(name)
                        pending.append(name)
            selected = [
                (path, cls, constants)
                for path, cls, constants in selected
                if cls.name in reachable
            ]
        elif selected:
            builder.warnings.append(
                "No model construction entry point resolved; showing source-defined "
                "NN classes, which may include unused alternatives."
            )
        for index, (path, cls, constants) in enumerate(selected):
            builder.parse_class(cls, index, path, constants)
            group = next(
                item for item in builder.groups if item["id"] == f"class-{index}"
            )
            group["is_entrypoint"] = cls.name in roots
            group["reachability"] = "source_reference" if roots else "unresolved"
        if ir is not None:
            builder.warnings.append(
                "Unrecognized or invalid IR schema; "
                "used static source inspection instead."
            )
        if not builder.nodes:
            builder.warnings.append(
                "No supported module declarations found; "
                "architecture structure is unavailable."
            )
        result = {
            "nodes": builder.nodes,
            "edges": builder.edges,
            "groups": builder.groups,
            "representation": "static_module_structure",
            "extraction": {
                "method": "python_ast",
                "completeness": "partial" if builder.nodes else "unavailable",
                "warnings": list(dict.fromkeys(builder.warnings)),
                "note": (
                    "Source-declared modules and selected explicit call dependencies. "
                    "Shapes, execution counts, helper internals and runtime topology "
                    "are unknown unless recorded. Class groups are definitions, "
                    "not expanded instances."
                ),
            },
        }
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "id": "architecture-" + digest[:20],
            "source_id": source_id,
            "source_hash": digest,
            "summary": {
                "nodes": len(result["nodes"]),
                "edges": len(result["edges"]),
                "kinds": dict(Counter(node["kind"] for node in result["nodes"])),
                "parameter_count": None,
            },
        }
    )
    return _safe(result)


_DIFF_FIELDS = (
    "kind",
    "operation",
    "config",
    "shape",
    "recurrence",
    "repeat",
    "is_output",
)


def _signature(node: Mapping[str, Any]) -> str:
    return _json({field: node.get(field) for field in _DIFF_FIELDS})


def diff_architectures(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Match structure, then stable ids; keep ambiguous matches explicit.

    ``added`` / ``removed`` contain node ids in the corresponding descriptor.
    ``changed`` contains before_id, after_id, changed fields, and their values.
    Edges are compared after translating matched before ids into after ids.
    """
    old = {node["id"]: node for node in before.get("nodes", [])}
    new = {node["id"]: node for node in after.get("nodes", [])}
    matches = []
    matched_old: set[str] = set()
    matched_new: set[str] = set()
    buckets: dict[str, list[str]] = defaultdict(list)
    for identifier, node in new.items():
        buckets[_signature(node)].append(identifier)
    for identifier, node in old.items():
        candidates = [
            item
            for item in buckets.get(_signature(node), [])
            if item not in matched_new
        ]
        if not candidates:
            continue
        same_group = [
            item
            for item in candidates
            if new[item].get("group_id") == node.get("group_id")
        ]
        candidates = same_group or candidates
        target = identifier if identifier in candidates else candidates[0]
        matches.append(
            {
                "before_id": identifier,
                "after_id": target,
                "reason": "structural_signature",
                "ambiguous": len(candidates) > 1,
            }
        )
        matched_old.add(identifier)
        matched_new.add(target)
    for identifier, node in old.items():
        if (
            identifier not in matched_old
            and identifier in new
            and identifier not in matched_new
            and node.get("kind") == new[identifier].get("kind")
        ):
            matches.append(
                {
                    "before_id": identifier,
                    "after_id": identifier,
                    "reason": "stable_declaration_id",
                    "ambiguous": False,
                }
            )
            matched_old.add(identifier)
            matched_new.add(identifier)
    changed = []
    for match in matches:
        left, right = old[match["before_id"]], new[match["after_id"]]
        fields = [
            field for field in _DIFF_FIELDS if left.get(field) != right.get(field)
        ]
        if fields:
            changed.append(
                {
                    "before_id": match["before_id"],
                    "after_id": match["after_id"],
                    "fields": fields,
                    "before": {field: left.get(field) for field in fields},
                    "after": {field: right.get(field) for field in fields},
                }
            )
    translation = {match["before_id"]: match["after_id"] for match in matches}

    def edge_key(
        edge: Mapping[str, Any], translate: bool = False
    ) -> tuple[str, str, str, str]:
        source, target = edge["source"], edge["target"]
        if translate:
            source, target = (
                translation.get(source, "old:" + source),
                translation.get(target, "old:" + target),
            )
        return (
            source,
            target,
            edge.get("type", "data_flow"),
            _json(
                {
                    "target_port": edge.get("target_port"),
                    "attributes": edge.get("attributes", {}),
                }
            ),
        )

    old_edges = {edge_key(edge, True): edge for edge in before.get("edges", [])}
    new_edges = {edge_key(edge): edge for edge in after.get("edges", [])}
    added = [identifier for identifier in new if identifier not in matched_new]
    removed = [identifier for identifier in old if identifier not in matched_old]
    added_edges = [edge for key, edge in new_edges.items() if key not in old_edges]
    removed_edges = [edge for key, edge in old_edges.items() if key not in new_edges]
    pieces = [
        f"{len(added)} added",
        f"{len(removed)} removed",
        f"{len(changed)} changed components",
        f"{len(added_edges)} added / {len(removed_edges)} removed connections",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "matches": matches,
        "added": added,
        "removed": removed,
        "changed": changed,
        "added_edges": added_edges,
        "removed_edges": removed_edges,
        "summary": {
            "added": len(added),
            "removed": len(removed),
            "changed": len(changed),
            "added_edges": len(added_edges),
            "removed_edges": len(removed_edges),
            "text": "; ".join(pieces),
        },
        "limitations": (
            "Compares available descriptors, not proof of semantic equivalence. "
            "Label-only renames are ignored; ambiguous duplicate matches are flagged."
        ),
    }
