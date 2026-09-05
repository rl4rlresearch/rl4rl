"""Four-digit v2.1 evaluator: editable representation and training budget.

No training ladder or paired-prefix search. Only data, fresh training accounting,
generic decoding, and exact-answer scoring are protected by this task adapter.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.c0c3_factorial import tiny_adderboard as data


def preflight_candidate_source(workspace: Path) -> str | None:
    path = workspace / "train.py"
    if not path.is_file() or path.is_symlink():
        return "train.py is missing or unsafe"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError) as error:
        return f"train.py does not compile: {error}"
    required = data.REQUIRED_FUNCTIONS | {
        "encode_inputs",
        "encode_targets",
        "decode_targets",
    }
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if required - functions:
        return f"missing interface functions: {sorted(required - functions)}"
    # Formatting arithmetic is permitted. The task's semantic learned-model
    # rule replaces v4's blanket ban on decimal operators and architecture text.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = {alias.name.split(".")[0] for alias in node.names}
            if imported & data.FORBIDDEN_IMPORTS:
                return "candidate imports a protected system module"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in data.FORBIDDEN_IMPORTS:
                return "candidate imports a protected system module"
        elif isinstance(node, ast.Call):
            name = data._dotted_name(node.func).lower()
            if name.rsplit(".", 1)[-1] in data.FORBIDDEN_CALLS:
                return f"candidate calls protected operation {name}"
            if name in data.FORBIDDEN_DOTTED_CALLS:
                return f"candidate calls protected operation {name}"
    return None


def _positive_setting(program, name: str) -> int:
    value = getattr(program, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _digits(numbers, width):
    return ((numbers[:, None] // data._powers(width)[None]) % 10).long()


def _token_matrix(value, rows):
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != rows:
        raise ValueError("codec must return a [batch, sequence] token matrix")
    if value.dtype != torch.long or value.shape[1] < 1 or bool((value < 0).any()):
        raise ValueError("codec must return nonnegative integer token IDs")
    return value


def _prompt(program, left, right):
    return _token_matrix(
        program.encode_inputs(_digits(left, 4), _digits(right, 4)), len(left)
    )


def _training_tensors(program, left, right):
    import torch

    prompt = _prompt(program, left, right)
    answers = _digits(left + right, 5)
    targets = _token_matrix(program.encode_targets(answers), len(left))
    if targets.shape[1] != _positive_setting(program, "OUTPUT_TOKENS"):
        raise ValueError("OUTPUT_TOKENS must match encoded target length")
    if not torch.equal(program.decode_targets(targets), answers):
        raise ValueError("target codec does not round-trip the supplied digits")
    full = torch.cat((prompt, targets), dim=1)
    inputs, labels = full[:, :-1].clone(), full[:, 1:].clone()
    labels[:, : prompt.shape[1] - 1] = -100
    return inputs, labels


def _generate(program, model, left, right, device):
    import torch

    sequence = _prompt(program, left, right).to(device)
    prefix = sequence.shape[1]
    for _ in range(_positive_setting(program, "OUTPUT_TOKENS")):
        logits = data._model_logits(model(sequence))
        if logits.ndim != 3 or logits.shape[:2] != sequence.shape:
            raise ValueError("model must return [batch, sequence, vocabulary] logits")
        sequence = torch.cat((sequence, logits[:, -1].argmax(-1, keepdim=True)), 1)
    # Codec sees generated answer tokens only; no input or label is passed.
    predicted = program.decode_targets(sequence[:, prefix:].cpu())
    if predicted.shape != (len(left), 5):
        raise ValueError("target decoder must return five digits per example")
    return predicted


def _score(program, model, left, right, device):
    import torch

    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(left), 2048):
            a, b = left[start : start + 2048], right[start : start + 2048]
            predicted = _generate(program, model, a, b, device)
            correct += int((predicted == _digits(a + b, 5)).all(1).sum())
    return correct / len(left)


def evaluate(args) -> int:
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    workspace = args.workspace.resolve()
    error = preflight_candidate_source(workspace)
    if error:
        raise ValueError(error)
    original_source = hashlib.sha256((workspace / "train.py").read_bytes()).hexdigest()
    program = data._load_program(workspace)
    steps = _positive_setting(program, "TRAINING_STEPS")
    batch_size = _positive_setting(program, "BATCH_SIZE")
    seed = int(os.environ.get("C0C3_RUN_SEED", args.seed)) % (2**63 - 1)
    data._seed_everything(seed)
    device = data._device(args.device)
    model = program.build_model().to(device)
    parameters = data._parameter_count(model)
    if parameters < 1:
        raise ValueError("model must have learned parameters")
    attention = data._attention_modules(model)
    if not attention or not any(list(module.parameters()) for module in attention):
        raise ValueError(
            "model must expose a learned module with Attention in its class name"
        )
    generator = torch.Generator().manual_seed(seed ^ 0x51A7E)
    public_a, public_b = data._sample_pairs(
        10000,
        generator=torch.Generator().manual_seed(seed ^ 0xA11CE),
        bucket_start=80,
        bucket_stop=90,
        unique=True,
    )
    optimizer = program.build_optimizer(model, steps)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("build_optimizer must return a torch optimizer")
    clip = float(getattr(program, "GRAD_CLIP_NORM", 1.0))
    started = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        left, right = data._sample_pairs(
            batch_size,
            generator=generator,
            bucket_start=0,
            bucket_stop=80,
            unique=False,
        )
        inputs, targets = _training_tensors(program, left, right)
        optimizer.zero_grad(set_to_none=True)
        loss = program.training_loss(
            model, inputs.to(device), targets.to(device), step, steps
        )
        if not isinstance(loss, torch.Tensor) or loss.ndim or not loss.isfinite():
            raise ValueError("training_loss must be a finite scalar tensor")
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        program.after_optimizer_step(optimizer, step, steps)
    data._synchronize(device)
    training_seconds = time.monotonic() - started
    if (
        original_source
        != hashlib.sha256((workspace / "train.py").read_bytes()).hexdigest()
    ):
        raise ValueError("source changed during training")
    if data._parameter_count(model) != parameters:
        raise ValueError("model parameter count changed during training")
    checkpoint = {
        "model": model.state_dict(),
        "step": steps,
        "parameters": parameters,
        "seed": seed,
        "source_sha256": original_source,
    }
    (workspace / "checkpoints").mkdir(exist_ok=True)
    for name in ["best.pt", "last.pt"]:
        torch.save(checkpoint, workspace / "checkpoints" / name)
    # Rebuild and reload the scored model, like the source-only v2.1 verifier.
    verified = program.build_model().to(device)
    verified.load_state_dict(
        torch.load(
            workspace / "checkpoints/best.pt", weights_only=True, map_location=device
        )["model"]
    )
    public_accuracy = _score(program, verified, public_a, public_b, device)
    accuracy = public_accuracy
    if args.layer == "C":
        left, right = data._sample_pairs(
            10000,
            generator=torch.Generator().manual_seed(seed ^ 0xC0FFEE),
            bucket_start=90,
            bucket_stop=100,
            unique=True,
        )
        accuracy = _score(program, verified, left, right, device)
    ablation_accuracy = None
    if public_accuracy >= 0.99:
        # An unused/dummy attention module cannot qualify a fixed solver.
        with torch.no_grad():
            for module in data._attention_modules(verified):
                for parameter in module.parameters():
                    parameter.zero_()
        ablation_accuracy = _score(
            program, verified, public_a[:512], public_b[:512], device
        )
        if ablation_accuracy > 0.2 or public_accuracy - ablation_accuracy < 0.5:
            raise ValueError("accuracy must depend meaningfully on learned attention")
    metrics = dict(
        accuracy=accuracy,
        parameters=parameters,
        training_steps=steps,
        training_seconds=training_seconds,
        attention_ablation_accuracy=ablation_accuracy,
    )
    result = dict(
        valid=accuracy >= 0.99,
        metrics=metrics,
        failure_kind=None if accuracy >= 0.99 else "nonqualification",
        protocol="tiny_adderboard_v21",
        device=str(device),
        source_sha256=original_source,
        seed=seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--layer", choices=["A", "C"], default="A")
    try:
        return evaluate(parser.parse_args())
    except ValueError as error:
        print(f"MODEL_CONTRACT_VIOLATION: {error}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
