"""Reproducible, conservative static source review for the ontology dashboard.

Never imports or executes candidate code. Positive AST evidence supplies partial
fingerprints; unresolved code changes remain uncertain, not preserving by default.
This is an exploratory source diagnostic, not blinded Layer B adjudication.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

VERSION = "source-review-v1"
CORE = "input_units input_transform embedding position mixing routing state feedforward parameter_construction sharing normalization connectivity aggregation output symmetry conditional_compute activation stochasticity bottleneck iteration other".split()
TASK_KEYS = {
    "addition": "operand_encoding digit_order carry_representation attention_scores projection_relations output_factorization",
    "nanogpt": "context_topology attention_scores kv_memory projection_relations vocabulary_representation block_composition",
    "fashion": "spatial_units spatial_operator scale_representation spatial_readout channel_interaction",
    "kws": "acoustic_representation state_update state_structure temporal_schedule readout_history exit_policy",
    "har": "sensor_fusion temporal_operator directionality state_update frequency_representation temporal_readout",
}
BOUNDARIES = {"mixing", "position", "routing", "state_update", "feedforward", "embedding",
              "aggregation", "spatial_operator", "conditional_compute", "exit_policy", "symmetry", "input_transform", "channel_interaction"}


def task_for(key):
    return "nanogpt" if "nanogpt" in key else "fashion" if "fashion" in key else "kws" if "kws" in key else "har" if "har" in key else "addition"


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + "." + node.attr
    return ""


def clean_ast(node):
    """Ignore comments/docstrings, but preserve operations, constants and control flow."""
    class Clean(ast.NodeTransformer):
        def visit_Expr(self, item):
            if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                return None
            return self.generic_visit(item)
    return ast.dump(Clean().visit(ast.parse(ast.unparse(node))), include_attributes=False)


def execution_shape(definitions, reached, module_classes):
    """Conservative comparison modulo affine/table/norm parameter coordinates.

    Only wrappers whose forward returns a standard primitive on the unmodified
    input qualify. Algebraic matrix products and trigonometric parameter
    generators do not qualify. Non-leaf forward/control flow is kept verbatim.
    """
    primitives = {"Linear": "affine", "Embedding": "lookup", "LayerNorm": "normalization",
                  "RMSNorm": "normalization", "BatchNorm1d": "normalization", "BatchNorm2d": "normalization"}
    leaf = {}
    for name in reached & module_classes:
        node = definitions[name]
        forward = next((m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "forward"), None)
        if not forward or len(forward.args.args) < 2:
            continue
        returns = [n for n in ast.walk(forward) if isinstance(n, ast.Return)]
        if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
            continue
        call = returns[0].value
        label = {"linear": "affine", "embedding": "lookup", "layer_norm": "normalization", "rms_norm": "normalization"}.get(dotted(call.func).split('.')[-1])
        if not label or not call.args or not isinstance(call.args[0], ast.Name) or call.args[0].id != forward.args.args[1].arg:
            continue
        if any(isinstance(n, ast.MatMult) or (isinstance(n, ast.Call) and re.search(r"(?:sin|cos|einsum|matmul)$", dotted(n.func))) for n in ast.walk(node)):
            continue
        leaf[name] = label
    shape = []
    for name in sorted(reached):
        node = definitions[name]
        if name in leaf:
            continue
        if isinstance(node, ast.ClassDef) and name in module_classes:
            for method in node.body:
                if not isinstance(method, ast.FunctionDef):
                    continue
                if method.name not in {"__init__", "setup_optimizer", "configure_optimizers", "init_weights", "_init_weights", "_initWeights", "_initialize_weights", "_set_from_full", "estimate_flops", "num_scaling_params", "generate"}:
                    shape.append((name, method.name, clean_ast(method)))
                elif method.name == "__init__":
                    for n in ast.walk(method):
                        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                            target = ",".join(ast.unparse(t) for t in n.targets)
                            if not target.startswith("self."):
                                continue
                            roles = []
                            for call in ast.walk(n.value):
                                if isinstance(call, ast.Call):
                                    cname = dotted(call.func).split('.')[-1]
                                    if cname in leaf or cname in module_classes or cname in primitives or cname in {"Conv1d", "Conv2d", "GRU", "LSTM", "RNN", "GELU", "ReLU", "SiLU", "MaxPool1d", "MaxPool2d", "AdaptiveAvgPool2d"}:
                                        roles.append(leaf.get(cname, primitives.get(cname, cname)))
                            if roles:
                                shape.append((name, target, tuple(sorted(set(roles)))))
                    # Preserve categorical switches/defaults even when dimensions change.
                    categorical = [n.value for n in ast.walk(method) if isinstance(n, ast.Constant) and isinstance(n.value, (str, bool))]
                    shape.append((name, "categorical configuration", categorical))
        else:
            shape.append((name, clean_ast(node)))
    return hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()


def analyze(sources, task):
    fp = dict.fromkeys(CORE + TASK_KEYS[task].split())
    evidence = {}
    trees = {}
    for filename, source in sources.items():
        try:
            trees[filename] = ast.parse(source)
        except SyntaxError:
            return {"fingerprint": fp, "error": "Source does not parse", "evidence": {}, "family": None}
    if not trees:
        return {"fingerprint": fp, "error": "Source artifacts missing", "evidence": {}, "family": None}
    definitions = {}
    locations = {}
    for filename, tree in trees.items():
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                definitions[node.name] = node
                locations[node.name] = filename
    module_classes = {name for name, n in definitions.items() if isinstance(n, ast.ClassDef)
                      and any(dotted(b).endswith(".Module") for b in n.bases)}
    for _ in range(3):
        module_classes.update(name for name, n in definitions.items() if isinstance(n, ast.ClassDef)
                              and any(dotted(b) in module_classes for b in n.bases))
    roots = set()
    builder = definitions.get("build_model")
    if builder:
        roots = {dotted(n.func) for n in ast.walk(builder) if isinstance(n, ast.Call)} & module_classes
    if not roots:
        roots = {name for name in ["GPT", "TinyDecoderLM"] if name in module_classes}
    if not roots:
        referenced = {n.id for name in module_classes for n in ast.walk(definitions[name])
                      if isinstance(n, ast.Name) and n.id != name}
        roots = module_classes - referenced
    if not roots:
        return {"fingerprint": fp, "error": "Model root cannot be resolved statically", "evidence": {}, "family": None}
    if len(roots) != 1:
        return {"fingerprint": fp, "error": "Multiple model factory branches require individual adjudication", "evidence": {}, "family": None}
    reached = set(roots)
    for _ in range(12):
        previous = set(reached)
        for name in previous:
            reached.update(n.id for n in ast.walk(definitions[name]) if isinstance(n, ast.Name) and n.id in definitions)
        if reached == previous:
            break
    # Keep inference/representation methods; training and initialization are separate.
    skip = {"setup_optimizer", "configure_optimizers", "init_weights", "_init_weights", "_initWeights", "_initialize_weights", "estimate_flops", "num_scaling_params", "generate"}
    pieces = []
    for name in sorted(reached):
        node = definitions[name]
        if isinstance(node, ast.ClassDef):
            if name not in module_classes:
                pieces.append((locations[name], node))  # Configuration classes matter.
            else:
                pieces.extend((locations[name], m) for m in node.body if isinstance(m, ast.FunctionDef) and m.name not in skip)
        else:
            pieces.append((locations[name], node))
    for name in ["encode_inputs", "encode_targets", "decode_targets", "build_model"]:
        if name in definitions and name not in reached:
            pieces.append((locations[name], definitions[name]))
    # Include all referenced global configuration assignments; do not execute them.
    names = {n.id for _, node in pieces for n in ast.walk(node) if isinstance(n, ast.Name)}
    for filename, tree in trees.items():
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id in names for t in targets):
                    pieces.append((filename, node))
    nodes = [(filename, n) for filename, node in pieces for n in ast.walk(node)]
    statement_text = [(f, n, ast.unparse(n)) for f, n in nodes
                      if isinstance(n, (ast.Assign, ast.AnnAssign, ast.Return))]
    calls = [(filename, n, dotted(n.func)) for filename, n in nodes if isinstance(n, ast.Call)]
    code = "\n".join(ast.unparse(n) for _, n in pieces)
    root_code = "\n".join(ast.unparse(m) for name in roots for m in definitions[name].body
                          if isinstance(m, ast.FunctionDef) and m.name in {"forward", "classify", "recurrent_step"})
    all_code = "\n".join(ast.unparse(t) for t in trees.values())

    def put(key, value, matches):
        fp[key] = value
        evidence[key] = [{"file": f, "line": n.lineno, "code": ast.unparse(n).splitlines()[0][:240]}
                         for f, n in matches[:3]]

    def matching(pattern, scope=calls):
        return [(f, n) for f, n, name in scope if re.search(pattern, name)]

    def text_matches(pattern):
        return [(f, n) for f, n, text in statement_text if re.search(pattern, text, re.I)]

    def primitive(pattern):
        return matching(r"(?:^|\.)" + pattern + r"$")

    recurrent = [(label, primitive(label + r"(?:Cell)?")) for label in ["GRU", "LSTM", "RNN"]]
    recurrent = [(label, hits) for label, hits in recurrent if hits]
    conv = [(label, primitive(label)) for label in ["Conv1d", "Conv2d", "Conv3d"]]
    conv = [(label, hits) for label, hits in conv if hits]
    attention = matching(r"scaled_dot_product_attention|flash_attn_func")
    attention += [(locations[name], m) for name in reached & module_classes
                  if "att" in name.lower() for m in definitions[name].body
                  if isinstance(m, ast.FunctionDef) and m.name == "forward"
                  and ("softmax" in ast.unparse(m) or "einsum" in ast.unparse(m))]
    custom_recurrence = [(locations[name], m) for name in roots for m in definitions[name].body
                         if isinstance(m, ast.FunctionDef) and m.name == "recurrent_step"]
    linear = primitive("Linear")
    operators = [label for label, _ in conv + recurrent] + (["attention"] if attention else [])
    if custom_recurrence and not recurrent:
        operators.append("custom recurrence")
    if operators:
        hits = [h for _, hs in conv + recurrent for h in hs] + attention + custom_recurrence
        put("mixing", " + ".join(operators), hits)
    elif linear:
        put("mixing", "dense affine network", linear)
    if recurrent:
        put("state", "+".join(label for label, _ in recurrent) + " recurrent state", recurrent[0][1])
        if task in {"kws", "har"}:
            put("state_update", "+".join(label for label, _ in recurrent), recurrent[0][1])
    elif custom_recurrence:
        put("state", "custom explicit recurrent state", custom_recurrence)
        if task in {"kws", "har"}:
            put("state_update", "custom recurrence", custom_recurrence)
    if linear:
        put("parameter_construction", "learned affine maps", linear)
    acts = []
    for label, pattern in [("GELU", "[Gg][Ee][Ll][Uu]"), ("ReLU", "[Rr][Ee][Ll][Uu]"), ("SiLU", "[Ss][Ii][Ll][Uu]"), ("tanh", "[Tt]anh"), ("sigmoid", "[Ss]igmoid")]:
        hits = primitive(pattern)
        if hits:
            acts.append(label)
    if acts:
        put("activation", " + ".join(acts), matching(r"gelu|relu|silu|tanh|sigmoid|GELU|ReLU|SiLU|Tanh|Sigmoid"))
    norms = [label for label in ["LayerNorm", "RMSNorm", "BatchNorm1d", "BatchNorm2d", "GroupNorm", "InstanceNorm2d"] if primitive(label)]
    if matching(r"rms_norm$"):
        norms.append("functional RMSNorm")
    if norms:
        put("normalization", " + ".join(norms), matching(r"Norm|rms_norm$"))
    dropout = matching(r"Dropout|dropout$")
    if dropout:
        put("stochasticity", "dropout declared (rate may be zero)", dropout)
    residual = text_matches(r"\b(?:x|hidden)\s*=\s*(?:x|hidden)\s*\+")
    if residual:
        put("connectivity", "residual additions", residual)
    embedding_assigns = [(f, n) for f, n in nodes if isinstance(n, ast.Assign)
                         and any(re.search(r"token.*emb|\bwte\b|word.*emb", ast.unparse(t), re.I) for t in n.targets)
                         and "Embedding(" in ast.unparse(n.value)]
    embedding_assigns += text_matches(r"['\"]wte['\"]:\s*nn\.Embedding")
    if embedding_assigns:
        factor = any("Sequential(" in ast.unparse(n) and "Linear(" in ast.unparse(n) for _, n in embedding_assigns)
        put("embedding", "factorized lookup table" if factor else "lookup table", embedding_assigns)
        if factor:
            put("bottleneck", "factorized token embedding", embedding_assigns)
    if task == "addition" and not embedding_assigns:
        token_types = set()
        for name in roots:
            for n in ast.walk(definitions[name]):
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and any(
                    re.search(r"token.*emb|emb.*token", ast.unparse(t), re.I) for t in n.targets
                ):
                    if dotted(n.value.func) in module_classes:
                        token_types.add(dotted(n.value.func))
        for _ in range(8):
            old = set(token_types)
            token_types.update(n.id for name in old for n in ast.walk(definitions[name])
                               if isinstance(n, ast.Name) and n.id in module_classes)
            if old == token_types:
                break
        token_lookups = [(locations[name], n) for name in token_types for n in ast.walk(definitions[name])
                         if isinstance(n, ast.Call) and re.search(r"(?:\.Embedding|\.embedding)$", dotted(n.func))]
        if token_lookups:
            put("embedding", "lookup table", token_lookups)
    bilinear = text_matches(r"interaction\s*=\s*\(left.unsqueeze\(-1\)\s*\*\s*right.unsqueeze\(-2\)\)")
    if task == "addition" and bilinear:
        put("embedding", "bilinear digit-composition", bilinear)
    # Audited corpus extensions: quotient token representations and algebraic coordinates.
    symmetric = text_matches(r"(?:low|min_digit|small)\s*=\s*torch.minimum\(left, right\)")
    if symmetric and "torch.maximum(left, right)" in code:
        put("symmetry", "unordered operand-pair quotient", symmetric)
        put("embedding", "unordered-pair lookup representation", symmetric)
    fixed_basis = text_matches(r"return .*@\s*self\.basis")
    if fixed_basis and "register_buffer('basis'" in code:
        put("parameter_construction", "learned coordinates with fixed algebraic basis", fixed_basis)
    rotary = matching(r"apply_rotary|rotary_emb|rotate_half")
    # Position construction routines alone do not prove rotary integration.
    rotary = [(f, n) for f, n in rotary if not dotted(n.func).endswith("_precompute_rotary_embeddings")]
    relative = text_matches(r"scores?\s*=.*relative|att\s*=.*relative|scores?\s*=.*rel_bias")
    additive = text_matches(r"\+.*(?:pos_emb|position_emb)|(?:pos_emb|position_emb).*\+")
    position = (["rotary"] if rotary else []) + (["relative score bias"] if relative else []) + (["additive"] if additive else [])
    if position:
        put("position", " + ".join(position), rotary + relative + additive)
    if attention:
        qk = text_matches(r"\bq\s*@\s*k\b|\bq\w*\s*@\s*k\w*\.|matmul\(q,.*k") + matching(r"scaled_dot_product_attention|flash_attn_func")
        if qk:
            put("routing", "content QK + position bias" if relative else "content QK", qk + relative)
        elif relative:
            put("routing", "position-only scores", relative)
        if task in {"addition", "nanogpt"} and fp["routing"]:
            put("attention_scores", fp["routing"], qk + relative)
    mlps = [(locations[name], m) for name in reached & module_classes
            if re.search("mlp|feedforward|swiglu", name, re.I) for m in definitions[name].body
            if isinstance(m, ast.FunctionDef) and m.name == "forward"]
    mlp_text = "\n".join(ast.unparse(m) for _, m in mlps)
    gated = bool(re.search(r"(?:silu|gelu|sigmoid|gate).*\*|\*.*(?:gate|up_proj)", mlp_text))
    if mlps:
        put("feedforward", "multiplicatively gated MLP" if gated else "ungated MLP", mlps)
    else:
        seqff = text_matches(r"self\.feedforward\s*=.*Sequential")
        if seqff:
            put("feedforward", "ungated MLP", seqff)
    aliases = [(f, n) for f, n in nodes if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Attribute) and n.value.attr == "weight"
               and any(isinstance(t, ast.Attribute) and t.attr == "weight" for t in n.targets)]
    if aliases:
        put("sharing", "explicit weight alias", aliases)
    adaptive_pool = matching(r"AdaptiveAvgPool[123]d$|adaptive_avg_pool[123]d$")
    flat = matching(r"Flatten$")
    root_mean = [(locations[name], n) for name in roots for m in definitions[name].body
                 if isinstance(m, ast.FunctionDef) and m.name in {"forward", "classify"}
                 for n in ast.walk(m) if isinstance(n, ast.Call) and dotted(n.func).endswith(".mean")]
    online_mean = text_matches(r"self\.classifier\(summary\s*/")
    learned_pool = [(locations[name], m) for name in roots for m in definitions[name].body
                    if isinstance(m, ast.FunctionDef) and m.name in {"forward", "classify"}
                    and re.search(r"(?:pool|attn|attention|score).*softmax|softmax.*(?:pool|attn|attention|score)", ast.unparse(m), re.I)]
    if online_mean:
        put("aggregation", "online mean of recurrent outputs", online_mean)
    elif learned_pool and task in {"kws", "har", "fashion"}:
        put("aggregation", "learned attention pooling", learned_pool)
    elif adaptive_pool:
        put("aggregation", "adaptive average pooling", adaptive_pool)
    elif flat:
        put("aggregation", "flattened spatial features", flat)
    elif root_mean and task in {"kws", "har", "fashion"} and not ("self.aggregation" in root_code):
        put("aggregation", "mean pooling", root_mean)
    multireadout = text_matches(r"classifier\(torch.cat\(\((?:mean_output|mean_hidden|mean_state),\s*(?:peak|max_output|max_hidden)")
    if multireadout:
        put("aggregation", "concatenated mean/max/terminal recurrent readout", multireadout)
    # A manually expanded GRU remains a GRU; introducing different state equations is separate.
    manual_gru = text_matches(r"return new_gate\s*\+\s*update_gate\s*\*\s*\(hidden\s*-\s*new_gate\)")
    single_gate = text_matches(r"hidden\s*=\s*retention\s*\*\s*hidden\s*\+\s*\(1(?:\.0)?\s*-\s*retention\)\s*\*\s*proposal")
    if task == "kws" and manual_gru and "reset_gate" in code:
        put("state_update", "GRU", manual_gru)
        put("mixing", "GRU", manual_gru)
        put("input_transform", "core features with candidate-only detail channel", manual_gru)
    elif task == "kws" and single_gate:
        put("state_update", "single-gate interpolation recurrence", single_gate)
        put("mixing", "single-gate recurrence", single_gate)
    if task == "har" and "self.aggregation" in root_code:
        # Multiple branches exist; label the declared scheme, not a guessed active path.
        put("aggregation", "configurable terminal/mean readout", text_matches(r"self\.aggregation\s*="))
    output = text_matches(r"self\.(?:classifier|lm_head|output|output_projection)\s*=.*Linear")
    if output:
        put("output", "learned linear readout", output)
    if task == "fashion":
        put("input_units", "grayscale image pixels", [(locations[next(iter(roots))], definitions[next(iter(roots))])])
        put("spatial_units", "image grid", conv[0][1] if conv else output)
        if conv:
            grouped = [(f, n) for f, n in conv[0][1] if any(k.arg == "groups" and not (isinstance(k.value, ast.Constant) and k.value.value == 1) for k in n.keywords)]
            put("spatial_operator", "grouped/depthwise Conv2d" if grouped else "dense Conv2d", grouped or conv[0][1])
        if fp["aggregation"]:
            put("spatial_readout", fp["aggregation"], [(locations[next(iter(roots))], definitions[next(iter(roots))])])
        channel_gates = []
        for name in reached & module_classes:
            for m in definitions[name].body:
                if not isinstance(m, ast.FunctionDef) or m.name != "forward":
                    continue
                tainted = {a.arg for a in m.args.args if a.arg != "self"}
                for n in ast.walk(m):
                    if isinstance(n, ast.Assign) and any(isinstance(v, ast.Name) and v.id in tainted for v in ast.walk(n.value)):
                        tainted.update(t.id for t in n.targets if isinstance(t, ast.Name))
                dynamic_sigmoid = any(isinstance(n, ast.Call) and dotted(n.func).endswith("sigmoid")
                                      and any(isinstance(v, ast.Name) and v.id in tainted for a in n.args for v in ast.walk(a))
                                      for n in ast.walk(m))
                if dynamic_sigmoid and re.search(r"return .*\*|\*.*(?:gate|weight|scale)", ast.unparse(m)):
                    channel_gates.append((locations[name], m))
        if channel_gates:
            put("channel_interaction", "input-dependent multiplicative feature gating", channel_gates)
        mixed_pool = text_matches(r"max_weight\s*\*\s*F.max_pool2d.*F.avg_pool2d")
        if mixed_pool:
            put("spatial_downsampling", "learned scalar max/average mixture", mixed_pool)
        elif primitive("MaxPool2d"):
            put("spatial_downsampling", "max pooling", primitive("MaxPool2d"))
    if task in {"har", "kws"}:
        put("input_units", "sensor time frames" if task == "har" else "acoustic feature frames", [(locations[next(iter(roots))], definitions[next(iter(roots))])])
        if task == "har":
            if fp["mixing"]:
                fp["temporal_operator"] = fp["mixing"]
            bidirectional = re.search(r"(?:bidirectional\s*=\s*True|['\"]bidirectional['\"]\s*:\s*True)", code)
            if bidirectional:
                put("directionality", "bidirectional recurrence declared", text_matches(r"bidirectional"))
            if fp["aggregation"]:
                fp["temporal_readout"] = fp["aggregation"]
            stats = text_matches(r"rawStats\s*=|sensorGroups\s*=|magnitudes\s*=")
            if stats and "rawStats" in root_code:
                put("input_transform", "sensor magnitudes and temporal statistics", stats)
                put("sensor_fusion", "sensor-group features plus learned cross-sensor mixing", stats)
            envelope = text_matches(r"envelope\s*=\s*torch.log1p\(power\)")
            if envelope:
                put("frequency_representation", "learned filter energy envelopes", envelope)
        else:
            schedules = [(locations[name], m) for name in roots for m in definitions[name].body if isinstance(m, ast.FunctionDef) and m.name == "frame_schedule"]
            if schedules:
                put("temporal_schedule", "static frame schedule", schedules)
            exits = [(locations[name], m) for name in roots for m in definitions[name].body if isinstance(m, ast.FunctionDef) and m.name == "exit_mask"]
            active_exits = [(f, n) for f, n in exits if not all(isinstance(r.value, (ast.Constant,)) and r.value.value is None for r in ast.walk(n) if isinstance(r, ast.Return))]
            if active_exits:
                put("exit_policy", "input-dependent exit hook", active_exits)
                put("conditional_compute", "adaptive stopping declared", active_exits)
            if fp["aggregation"]:
                fp["readout_history"] = fp["aggregation"]
    if task == "addition":
        pairs = text_matches(r"left_digits\s*\*\s*10\s*\+\s*right_digits")
        if pairs:
            put("operand_encoding", "paired digit tokens", pairs)
            put("input_units", "operand-pair tokens", pairs)
        elif "TinyDecoderLM" in roots:
            put("input_units", "token sequence", [(locations["TinyDecoderLM"], definitions["TinyDecoderLM"])])
    if task == "nanogpt":
        put("input_units", "language token sequence", [(locations[next(iter(roots))], definitions[next(iter(roots))])])
        windows = text_matches(r"window_sizes|window_size=")
        if windows:
            put("context_topology", "configured local/global attention windows", windows)
        ve = text_matches(r"v\s*=\s*v\s*\+.*ve")
        if ve:
            put("kv_memory", "token value embeddings added to V", ve)
    training = {}
    inference = {}
    all_calls = [(filename, n, dotted(n.func)) for filename, t in trees.items() for n in ast.walk(t) if isinstance(n, ast.Call)]
    opt = sorted({name.split('.')[-1] for _, _, name in all_calls if re.search(r"(?:AdamW|Adam|SGD|RMSprop|MuonAdamW)$", name)})
    if opt:
        training["optimizer"] = " + ".join(opt) + " (declared)"
    if "cross_entropy" in all_code:
        training["objective"] = "cross entropy declared"
    if re.search(r"flip\(|roll\(|affine_grid|grid_sample", all_code):
        training["augmentation"] = "geometric transformations declared"
    if "torch.compile" in all_code:
        inference["execution"] = "compiled execution declared"
    settings = {}
    for _, tree in trees.items():
        for n in tree.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for target in targets:
                    if isinstance(target, ast.Name) and re.search(r"WIDTH|HIDDEN|DEPTH|LAYER|HEAD|RANK|BATCH|STEP|LR|RATE|CONFIG", target.id, re.I):
                        try:
                            value = ast.literal_eval(n.value)
                            if isinstance(value, (int, float, str, bool, dict, list, tuple)):
                                settings[target.id] = value
                        except (ValueError, TypeError):
                            pass
    signatures = {k: v for k, v in fp.items() if k in BOUNDARIES and v is not None}
    # Factorizing an embedding table is not a change from table to function.
    if signatures.get("embedding") == "factorized lookup table":
        signatures["embedding"] = "lookup table"
    digest = hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest()[:8]
    family = (fp["mixing"] or "partial architecture") + " · " + digest
    ast_digest = hashlib.sha256("\n".join(f+clean_ast(n) for f, n in pieces).encode()).hexdigest()
    return {"fingerprint": fp, "evidence": evidence, "family": family, "family_signature": signatures,
            "architecture_ast": ast_digest, "training": training, "inference": inference,
            "execution_shape": execution_shape(definitions, reached, module_classes),
            "settings": settings, "roots": sorted(roots), "error": None}


def read_sources(path):
    sources = {}
    names = ["src/model.py", "src/train.py"] if (path / "src").is_dir() else ["model.py", "train.py"]
    for name in names:
        try:
            text = (path / name).read_text(encoding="utf-8-sig")
        except OSError:
            continue
        if len(text) <= 2_000_000:
            sources[name] = text
    return sources


def analyze_job(job):
    sha, sources, task = job
    return sha, analyze(sources, task)


def review(snapshot_path, output, campaign_filter=None):
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key, campaign in snapshot["campaigns"].items():
        if not campaign.get("available"):
            continue
        if campaign_filter and key not in campaign_filter:
            continue
        task = task_for(key)
        paths = {}
        points = []
        for run in campaign["runs"]:
            for point in run["points"]:
                cid = point.get("candidate_id")
                if not cid:
                    continue
                rel = point.get("artifact_path") or "candidates/" + cid
                path = Path(campaign["campaign"]) / "runs" / run["run_id"] / rel.replace("\\", "/")
                paths[str(path)] = path
                points.append((run, point, str(path)))
                for parent in point.get("parent_ids") or []:
                    parentpath = Path(campaign["campaign"]) / "runs" / run["run_id"] / "candidates" / parent
                    paths[str(parentpath)] = parentpath
        print(f"{key}: reading {len(paths)} artifacts", flush=True)
        profiles = {}
        cache = {}
        source_hashes = {}
        source_bundles = {}
        with ThreadPoolExecutor(max_workers=12) as pool:
            for index, (path, sources) in enumerate(zip(paths, pool.map(read_sources, paths.values()))):
                sha = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
                source_bundles.setdefault(sha, sources)
                source_hashes[path] = sha if sources else None
        print(f"{key}: reviewing {len(source_bundles)} unique sources with four CPU workers", flush=True)
        with ProcessPoolExecutor(max_workers=4) as pool:
            jobs = ((sha, sources, task) for sha, sources in source_bundles.items())
            for index, (sha, result) in enumerate(pool.map(analyze_job, jobs, chunksize=8)):
                cache[sha] = result
                if index and index % 500 == 0:
                    print(f"{key}: analyzed {index}/{len(source_bundles)} unique sources", flush=True)
        empty_hash = hashlib.sha256(b"{}").hexdigest()
        for path in paths:
            profiles[path] = cache[source_hashes[path] or empty_hash]
        rows = []
        examples = {}
        for run, point, path in points:
            profile = profiles[path]
            parents = [str(Path(campaign["campaign"]) / "runs" / run["run_id"] / "candidates" / cid) for cid in point.get("parent_ids") or []]
            parent = profiles.get(parents[0]) if len(parents) == 1 else None
            seed = point.get("is_seed", False)
            error = profile.get("error")
            same_source = bool(parents and source_hashes[path] and source_hashes[path] == source_hashes.get(parents[0]))
            implemented = None if error else not same_source
            if seed:
                implemented = not error
            changed = []
            classification = "uncertain"
            if error:
                classification = "unannotated"
            elif seed or same_source:
                classification = "preserving"
            elif parent and not parent.get("error"):
                before, after = parent["family_signature"], profile["family_signature"]
                changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
                if changed:
                    classification = "changing"
                elif profile["architecture_ast"] == parent["architecture_ast"]:
                    classification = "preserving"
                elif profile.get("execution_shape") and profile["execution_shape"] == parent.get("execution_shape"):
                    classification = "preserving"
            notes = error or ("Seed reference fingerprint." if seed else "No source change from recorded parent." if same_source else
                             "Static source evidence changes: " + ", ".join(changed) + "." if changed else
                             "Model execution shape unchanged modulo affine/table/normalization coordinate wrappers, numerical module settings and training code." if classification == "preserving" else
                             "Model source differs, but the partial fingerprint does not establish whether a representational boundary was crossed.")
            notes += " Partial static source review; category presence is not proof of runtime use. Unsupported components are null. No performance or condition used to assign mechanism categories."
            row = {"run_id": run["run_id"], "proposal": point["proposal"], "candidate_id": point["candidate_id"],
                   "family": profile["family"] if classification in {"preserving", "changing"} else None, "classification": classification,
                   "proposed_change": None, "implemented": implemented,
                   "executable": True if point.get("valid") or point.get("failure_kind") == "nonqualification" else None,
                   "fingerprint": profile["fingerprint"], "training": profile.get("training", {}),
                   "inference": profile.get("inference", {}), "settings": profile.get("settings", {}),
                   "reviewer": "Codex · audited static source rules v1", "notes": notes,
                   "source_sha256": source_hashes[path], "component_evidence": profile["evidence"],
                   "changed_components": changed, "review_method": "static AST extraction with conservative unresolved labels"}
            rows.append(row)
            if row["family"]:
                examples.setdefault(profile["family"], {"path": path, "fingerprint": profile["fingerprint"], "signature": profile["family_signature"], "count": 0})["count"] += 1
        doc = {"schema_version": "1.0", "campaign": campaign["campaign"], "rubric": VERSION + " · partial computational fingerprints; unresolved source edits remain uncertain",
               "generated_at": datetime.now(timezone.utc).isoformat(), "snapshot_generated_at": snapshot.get("generated_at"),
               "task": task, "method": "Source-only static review; seed and representative family evidence audited by Codex. Not blinded or runtime-validated.", "rows": rows}
        (output / (key + ".json")).write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        (output / (key + "-families.json")).write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
        summary[key] = {"rows": len(rows), "classes": dict(collections.Counter(r["classification"] for r in rows)), "families": len(examples),
                        "components": dict(collections.Counter(k for r in rows for k,v in r["fingerprint"].items() if v is not None))}
        print(key, summary[key]["classes"], "families", len(examples), flush=True)
    (output / "review-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=Path("outputs/ontology/dashboard-snapshot.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ontology"))
    parser.add_argument("--campaign", action="append")
    args = parser.parse_args()
    review(args.snapshot, args.output, args.campaign)
