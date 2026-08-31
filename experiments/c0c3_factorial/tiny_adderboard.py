#!/usr/bin/env python3
"""Protected evaluator for a compute-matched four-digit addition task."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

NUM_DIGITS = 4
SUM_DIGITS = 5
MAX_OPERAND = 10**NUM_DIGITS
DIGIT_COUNT = 10
PAIR_COUNT = 100
PAIR_BASE = DIGIT_COUNT
EQUALS_ID = PAIR_BASE + PAIR_COUNT
BOS_ID = EQUALS_ID + 1
EOS_ID = BOS_ID + 1
PAD_ID = EOS_ID + 1
VOCAB_SIZE = PAD_ID + 1
PROMPT_LENGTH = NUM_DIGITS + 2
TARGET_LENGTH = SUM_DIGITS + 1
INPUT_LENGTH = PROMPT_LENGTH + TARGET_LENGTH - 1
TRAIN_BUCKET_STOP = 80
PUBLIC_BUCKET_START = 80
PUBLIC_BUCKET_STOP = 90
HOLDOUT_BUCKET_START = 90
HOLDOUT_BUCKET_STOP = 100
DEFAULT_LADDER = [200, 400, 600, 1_000]

FORBIDDEN_IMPORTS = {
    "ctypes",
    "importlib",
    "inspect",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
FORBIDDEN_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "open",
}
FORBIDDEN_DOTTED_CALLS = {
    "torch.load",
    "torch.save",
}
FORBIDDEN_SOURCE_PATTERNS = {
    "carry-specific model logic": "carry",
    "finite-state arithmetic": "transducer",
    "fixed decimal remainder": "% 10",
    "fixed decimal quotient": "// 10",
    "hidden answer table": "lookup_table",
}
REQUIRED_FUNCTIONS = {
    "after_optimizer_step",
    "build_model",
    "build_optimizer",
    "training_loss",
}


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def preflight_candidate_source(workspace: Path) -> str | None:
    """Reject source that can escape the protected learned-model interface."""

    path = workspace / "train.py"
    if not path.is_file() or path.is_symlink():
        return "train.py is missing or unsafe"
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename="train.py")
    except (OSError, SyntaxError, UnicodeError) as error:
        return f"train.py does not compile: {error}"
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    missing = sorted(REQUIRED_FUNCTIONS - defined)
    if missing:
        return f"train.py is missing required functions: {', '.join(missing)}"
    normalized = source.casefold().replace(" ", "")
    for description, text in FORBIDDEN_SOURCE_PATTERNS.items():
        if text.replace(" ", "") in normalized:
            return f"train.py contains {description}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = {alias.name.split(".", 1)[0] for alias in node.names}
            forbidden = sorted(imported & FORBIDDEN_IMPORTS)
            if forbidden:
                return f"train.py imports protected module {forbidden[0]}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in FORBIDDEN_IMPORTS:
                return f"train.py imports protected module {root}"
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func).lower()
            if name.rsplit(".", 1)[-1] in FORBIDDEN_CALLS:
                return f"train.py calls protected operation {name}"
            if name in FORBIDDEN_DOTTED_CALLS:
                return f"train.py calls protected operation {name}"
    return None


def _literal_assignment(path: Path, symbol: str) -> object | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return None
    value: object | None = None
    for node in tree.body:
        targets: list[ast.expr] = []
        expression: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            expression = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            expression = node.value
        if expression is None or not any(
            isinstance(target, ast.Name) and target.id == symbol
            for target in targets
        ):
            continue
        try:
            value = ast.literal_eval(expression)
        except (ValueError, TypeError):
            return None
    return value


def _resolved_ladder(
    workspace: Path, args: argparse.Namespace
) -> tuple[list[int], dict[str, Any]]:
    defaults = sorted(set(int(value) for value in args.ladder_levels))
    terminal = int(args.max_steps)
    if terminal not in defaults:
        defaults.append(terminal)
        defaults.sort()
    receipt: dict[str, Any] = {
        "source": "controller_default",
        "accepted": False,
        "reason": "candidate policy absent",
        "levels": defaults,
    }
    raw = _literal_assignment(workspace / "train.py", "EVALUATION_LADDER")
    if not isinstance(raw, list | tuple):
        return defaults, receipt
    try:
        levels = [int(value) for value in raw]
    except (TypeError, ValueError):
        receipt["reason"] = "EVALUATION_LADDER is not an integer sequence"
        return defaults, receipt
    if terminal not in levels:
        levels.append(terminal)
    if (
        levels != sorted(set(levels))
        or any(value < args.ladder_minimum or value > terminal for value in levels)
        or len(levels) > args.ladder_max_rungs
    ):
        receipt["reason"] = "candidate ladder exceeded evaluator-owned bounds"
        return defaults, receipt
    receipt.update(
        {
            "source": "train.py",
            "accepted": True,
            "reason": "safe literal candidate policy accepted",
            "levels": levels,
            "enforced_minimum_level": args.ladder_minimum,
            "enforced_terminal_level": terminal,
            "enforced_max_rungs": args.ladder_max_rungs,
        }
    )
    return levels, receipt


def _load_program(workspace: Path):
    path = workspace / "train.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    name = f"tiny_adderboard_candidate_{digest}"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _device(name: str):
    import torch

    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Tiny AdderBoard requires an available MPS device")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA device is unavailable")
    return torch.device(name)


def _synchronize(device) -> None:
    import torch

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    import torch

    torch.manual_seed(seed)


def _split_bucket(a, b):
    """Stable vectorized split hash; train/public/holdout buckets never overlap."""

    return (a * 17 + b * 31 + (a ^ b) * 13 + (a * b) * 7) % 100


def _sample_pairs(
    count: int,
    *,
    generator,
    bucket_start: int,
    bucket_stop: int,
    unique: bool,
):
    import torch

    if not unique:
        a = torch.randint(
            0, MAX_OPERAND, (count,), generator=generator, dtype=torch.int64
        )
        b = torch.randint(
            0, MAX_OPERAND, (count,), generator=generator, dtype=torch.int64
        )
        eligible = (_split_bucket(a, b) >= bucket_start) & (
            _split_bucket(a, b) < bucket_stop
        )
        while not bool(eligible.all()):
            missing = int((~eligible).sum())
            a[~eligible] = torch.randint(
                0, MAX_OPERAND, (missing,), generator=generator, dtype=torch.int64
            )
            b[~eligible] = torch.randint(
                0, MAX_OPERAND, (missing,), generator=generator, dtype=torch.int64
            )
            eligible = (_split_bucket(a, b) >= bucket_start) & (
                _split_bucket(a, b) < bucket_stop
            )
        return a, b

    chunks_a: list[torch.Tensor] = []
    chunks_b: list[torch.Tensor] = []
    seen: set[int] = set()
    collected = 0
    while collected < count:
        sample_count = max(2_048, (count - collected) * 3)
        a = torch.randint(
            0, MAX_OPERAND, (sample_count,), generator=generator, dtype=torch.int64
        )
        b = torch.randint(
            0, MAX_OPERAND, (sample_count,), generator=generator, dtype=torch.int64
        )
        mask = (_split_bucket(a, b) >= bucket_start) & (
            _split_bucket(a, b) < bucket_stop
        )
        a = a[mask]
        b = b[mask]
        if unique:
            keep_a: list[int] = []
            keep_b: list[int] = []
            for left, right in zip(a.tolist(), b.tolist(), strict=True):
                key = left * MAX_OPERAND + right
                if key in seen:
                    continue
                seen.add(key)
                keep_a.append(left)
                keep_b.append(right)
            a = torch.tensor(keep_a, dtype=torch.int64)
            b = torch.tensor(keep_b, dtype=torch.int64)
        needed = count - collected
        chunks_a.append(a[:needed])
        chunks_b.append(b[:needed])
        collected += min(needed, len(a))
    return torch.cat(chunks_a), torch.cat(chunks_b)


def _powers(width: int):
    import torch

    return torch.tensor([10**index for index in range(width)], dtype=torch.int64)


def _prompt_tokens(a, b):
    import torch

    powers = _powers(NUM_DIGITS)
    a_digits = ((a[:, None] // powers[None, :]) % 10).to(torch.long)
    b_digits = ((b[:, None] // powers[None, :]) % 10).to(torch.long)
    pairs = PAIR_BASE + a_digits * 10 + b_digits
    bos = torch.full((len(a), 1), BOS_ID, dtype=torch.long)
    equals = torch.full((len(a), 1), EQUALS_ID, dtype=torch.long)
    return torch.cat((bos, pairs, equals), dim=1)


def _training_tensors(a, b):
    import torch

    prompt = _prompt_tokens(a, b)
    powers = _powers(SUM_DIGITS)
    answer = ((a[:, None] + b[:, None]) // powers[None, :]) % 10
    answer = answer.to(torch.long)
    eos = torch.full((len(a), 1), EOS_ID, dtype=torch.long)
    full = torch.cat((prompt, answer, eos), dim=1)
    token_ids = full[:, :-1].clone()
    targets = full[:, 1:].clone()
    targets[:, : PROMPT_LENGTH - 1] = -100
    return token_ids, targets


def _model_logits(output):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ValueError("model must return token logits or a tuple beginning with logits")


def _generate(model, a, b, *, device):
    import torch

    generated = _prompt_tokens(a, b).to(device)
    for _ in range(TARGET_LENGTH):
        logits = _model_logits(model(generated))
        if logits.ndim != 3 or logits.shape[:2] != generated.shape:
            raise ValueError("model must return [batch, sequence, vocabulary] logits")
        if logits.shape[-1] != VOCAB_SIZE:
            raise ValueError(f"model vocabulary must contain {VOCAB_SIZE} tokens")
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    return generated[:, PROMPT_LENGTH : PROMPT_LENGTH + SUM_DIGITS].cpu()


def _score(model, a, b, *, batch_size: int, device) -> dict[str, int | float]:
    import torch

    model.eval()
    correct = 0
    token_correct = 0
    powers = _powers(SUM_DIGITS)
    with torch.no_grad():
        for start in range(0, len(a), batch_size):
            left = a[start : start + batch_size]
            right = b[start : start + batch_size]
            predicted = _generate(model, left, right, device=device)
            expected = ((left[:, None] + right[:, None]) // powers[None, :]) % 10
            matches = predicted.eq(expected.to(torch.long))
            token_correct += int(matches.sum())
            correct += int(matches.all(dim=1).sum())
    cases = len(a)
    return {
        "accuracy": correct / cases,
        "correct": correct,
        "cases": cases,
        "digit_accuracy": token_correct / (cases * SUM_DIGITS),
    }


def _attention_modules(model) -> list[Any]:
    import torch

    return [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.MultiheadAttention)
        or "attention" in module.__class__.__name__.casefold()
    ]


def _attention_contract(model, probe_a, probe_b, *, device) -> str | None:
    import torch

    modules = _attention_modules(model)
    if not modules:
        return "the model has no learned self-attention module"
    fired: set[int] = set()
    handles = [
        module.register_forward_hook(
            lambda current, _inputs, _output: fired.add(id(current))
        )
        for module in modules
    ]
    try:
        with torch.no_grad():
            _generate(model, probe_a[:8], probe_b[:8], device=device)
    finally:
        for handle in handles:
            handle.remove()
    if not fired:
        return "no learned self-attention module participates in the forward pass"
    parameters = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    if not parameters or sum(parameter.numel() for parameter in parameters) < 1:
        return "the attention path has no learned parameters"
    return None


def _attention_ablation_accuracy(model, a, b, *, batch_size: int, device) -> float:
    import torch

    parameters = []
    seen: set[int] = set()
    for module in _attention_modules(model):
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    saved = [parameter.detach().clone() for parameter in parameters]
    try:
        with torch.no_grad():
            for parameter in parameters:
                parameter.zero_()
        score = _score(model, a, b, batch_size=batch_size, device=device)
        return float(score["accuracy"])
    finally:
        with torch.no_grad():
            for parameter, value in zip(parameters, saved, strict=True):
                parameter.copy_(value)


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def evaluate(args: argparse.Namespace) -> int:
    import torch

    workspace = args.workspace.resolve()
    source_error = preflight_candidate_source(workspace)
    if source_error is not None:
        print(f"MODEL_CONTRACT_VIOLATION: {source_error}")
        return 3
    program = _load_program(workspace)
    levels, policy = _resolved_ladder(workspace, args)
    seed = int(os.environ.get("C0C3_RUN_SEED", args.seed)) % (2**63 - 1)
    _seed_everything(seed)
    device = _device(args.device)
    model = program.build_model().to(device)
    parameters = _parameter_count(model)
    if parameters < 1 or parameters > args.max_parameters:
        print(
            "MODEL_CONTRACT_VIOLATION: learned parameter count "
            f"{parameters} is outside [1, {args.max_parameters}]"
        )
        return 3
    generator = torch.Generator().manual_seed(seed ^ 0x51A7E)
    validation_generator = torch.Generator().manual_seed(seed ^ 0xA11CE)
    validation_a, validation_b = _sample_pairs(
        args.validation_cases,
        generator=validation_generator,
        bucket_start=PUBLIC_BUCKET_START,
        bucket_stop=PUBLIC_BUCKET_STOP,
        unique=True,
    )
    holdout_a = holdout_b = None
    if args.layer == "C":
        holdout_generator = torch.Generator().manual_seed(seed ^ 0xC0FFEE)
        holdout_a, holdout_b = _sample_pairs(
            args.holdout_cases,
            generator=holdout_generator,
            bucket_start=HOLDOUT_BUCKET_START,
            bucket_stop=HOLDOUT_BUCKET_STOP,
            unique=True,
        )
    attention_error = _attention_contract(
        model, validation_a, validation_b, device=device
    )
    if attention_error is not None:
        print(f"MODEL_CONTRACT_VIOLATION: {attention_error}")
        return 3
    batch_size = getattr(program, "BATCH_SIZE", 512)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        print("MODEL_CONTRACT_VIOLATION: BATCH_SIZE must be an integer")
        return 3
    if not args.minimum_batch_size <= batch_size <= args.maximum_batch_size:
        print(
            "MODEL_CONTRACT_VIOLATION: BATCH_SIZE is outside the evaluator-owned "
            f"range [{args.minimum_batch_size}, {args.maximum_batch_size}]"
        )
        return 3
    optimizer = program.build_optimizer(model, levels[-1])
    if not isinstance(optimizer, torch.optim.Optimizer):
        print("MODEL_CONTRACT_VIOLATION: build_optimizer returned an invalid optimizer")
        return 3
    grad_clip = float(getattr(program, "GRAD_CLIP_NORM", 1.0))
    if not 0.0 <= grad_clip <= 100.0:
        print("MODEL_CONTRACT_VIOLATION: GRAD_CLIP_NORM is outside [0, 100]")
        return 3
    stages: list[dict[str, Any]] = []
    qualified_step: int | None = None
    final_public: dict[str, int | float] | None = None
    _synchronize(device)
    started = time.monotonic()
    for step in range(1, levels[-1] + 1):
        model.train()
        left, right = _sample_pairs(
            batch_size,
            generator=generator,
            bucket_start=0,
            bucket_stop=TRAIN_BUCKET_STOP,
            unique=False,
        )
        token_ids, targets = _training_tensors(left, right)
        token_ids = token_ids.to(device)
        targets = targets.to(device)
        loss = program.training_loss(model, token_ids, targets, step, levels[-1])
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.isfinite():
            print("MODEL_CONTRACT_VIOLATION: training_loss returned an invalid scalar")
            return 3
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        program.after_optimizer_step(optimizer, step, levels[-1])
        if step not in levels:
            continue
        public_score = _score(
            model,
            validation_a,
            validation_b,
            batch_size=args.evaluation_batch_size,
            device=device,
        )
        final_public = public_score
        promoted = float(public_score["accuracy"]) >= args.qualification_minimum
        stages.append(
            {
                "stage": len(stages) + 1,
                "training_steps": step,
                "accuracy": public_score["accuracy"],
                "correct": public_score["correct"],
                "cases": public_score["cases"],
                "qualified": promoted,
            }
        )
        if promoted:
            qualified_step = step
            break
    _synchronize(device)
    training_seconds = time.monotonic() - started
    if final_public is None:
        raise RuntimeError("the evaluator reached no training ladder rung")
    if args.layer == "C":
        assert holdout_a is not None and holdout_b is not None
        reported = _score(
            model,
            holdout_a,
            holdout_b,
            batch_size=args.evaluation_batch_size,
            device=device,
        )
        split_name = "sealed_holdout"
    else:
        reported = final_public
        split_name = "public_validation"
    valid = (
        qualified_step is not None
        and float(reported["accuracy"]) >= args.qualification_minimum
    )
    ablation_accuracy: float | None = None
    if valid:
        ablation_cases = min(args.ablation_cases, len(validation_a))
        ablation_accuracy = _attention_ablation_accuracy(
            model,
            validation_a[:ablation_cases],
            validation_b[:ablation_cases],
            batch_size=args.evaluation_batch_size,
            device=device,
        )
        if (
            ablation_accuracy > args.maximum_ablation_accuracy
            or float(final_public["accuracy"]) - ablation_accuracy
            < args.minimum_attention_accuracy_drop
        ):
            print(
                "MODEL_CONTRACT_VIOLATION: exact accuracy does not depend enough "
                "on learned self-attention"
            )
            return 3
    metrics: dict[str, Any] = {
        **reported,
        "parameters": parameters,
        "training_steps": qualified_step or levels[-1],
        "optimizer_steps": qualified_step or levels[-1],
        "training_seconds": training_seconds,
        "evaluation_split": split_name,
        "data_partition": "hash_disjoint_train_public_holdout_v1",
        "fidelity_highest_level": qualified_step or levels[-1],
        "fidelity_qualification_level": qualified_step,
        "fidelity_reached_full": (qualified_step or levels[-1]) == levels[-1],
        "fidelity_stage_count": len(stages),
        "fidelity_stages": stages,
        "fidelity_policy": policy,
        "attention_ablation_accuracy": ablation_accuracy,
    }
    payload = {
        "schema_version": "1.0",
        "layer": args.layer,
        "valid": valid,
        "failure_kind": None if valid else "nonqualification",
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "Tiny AdderBoard: "
        f"{reported['correct']}/{reported['cases']} correct "
        f"({100 * float(reported['accuracy']):.4f}%), "
        f"parameters={parameters}, steps={metrics['training_steps']}, "
        f"seconds={training_seconds:.3f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, layer in (("evaluate-ladder", "A"), ("holdout", "C")):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--workspace", type=Path, required=True)
        subparser.add_argument("--repo-root", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)
        subparser.add_argument(
            "--device", choices=("cpu", "mps", "cuda"), default="cpu"
        )
        subparser.add_argument("--seed", type=int, default=20260828)
        subparser.add_argument("--max-steps", type=int, default=1_000)
        subparser.add_argument(
            "--ladder-levels", type=int, nargs="+", default=DEFAULT_LADDER
        )
        subparser.add_argument("--ladder-minimum", type=int, default=100)
        subparser.add_argument("--ladder-max-rungs", type=int, default=6)
        subparser.add_argument("--qualification-minimum", type=float, default=0.99)
        subparser.add_argument("--validation-cases", type=int, default=10_000)
        subparser.add_argument("--holdout-cases", type=int, default=10_000)
        subparser.add_argument("--evaluation-batch-size", type=int, default=512)
        subparser.add_argument("--minimum-batch-size", type=int, default=32)
        subparser.add_argument("--maximum-batch-size", type=int, default=2_048)
        subparser.add_argument("--max-parameters", type=int, default=25_000)
        subparser.add_argument("--ablation-cases", type=int, default=1_000)
        subparser.add_argument("--maximum-ablation-accuracy", type=float, default=0.25)
        subparser.add_argument(
            "--minimum-attention-accuracy-drop", type=float, default=0.50
        )
        subparser.add_argument("--verify-existing-checkpoint", action="store_true")
        subparser.set_defaults(layer=layer, handler=evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    del args.repo_root, args.verify_existing_checkpoint
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
