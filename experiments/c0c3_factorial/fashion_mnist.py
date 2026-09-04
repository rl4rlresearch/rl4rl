#!/usr/bin/env python3
"""Protected fixed-exposure evaluator for the Fashion-MNIST task stratum."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import struct
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

DATA_ROOT_ENV = "RL4RL_FASHION_MNIST_DATA_ROOT"
SPLIT_SEED = 20_260_823
TRAIN_EXAMPLES = 50_000
VALIDATION_EXAMPLES = 10_000
DEFAULT_TRAINING_EXPOSURE = 100_000
DEFAULT_MAX_PARAMETERS = 250_000
NORMALIZATION_MEAN = 0.2860406
NORMALIZATION_STD = 0.35302424
FORBIDDEN_CANDIDATE_IMPORTS = frozenset(
    {
        "glob",
        "http",
        "marshal",
        "os",
        "pathlib",
        "pickle",
        "requests",
        "shelve",
        "shutil",
        "socket",
        "sqlite3",
        "subprocess",
        "torchvision",
        "urllib",
    }
)
FORBIDDEN_CANDIDATE_CALLS = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "open",
    }
)
FORBIDDEN_CANDIDATE_DOTTED_CALLS = frozenset(
    {
        "np.fromfile",
        "np.genfromtxt",
        "np.load",
        "np.loadtxt",
        "numpy.fromfile",
        "numpy.genfromtxt",
        "numpy.load",
        "numpy.loadtxt",
        "torch.hub.load",
        "torch.load",
    }
)

# The checksums published by the Fashion-MNIST project identify the immutable
# IDX payloads.  Downloads are an explicit setup operation; evaluation never
# accesses the network.
DATA_FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
DOWNLOAD_BASE = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
    "master/data/fashion"
)


def data_root(repo_root: Path, configured: Path | None = None) -> Path:
    if configured is not None:
        return configured.expanduser().resolve()
    environment = os.environ.get(DATA_ROOT_ENV)
    if environment:
        return Path(environment).expanduser().resolve()
    return (repo_root / "data/raw/fashion-mnist").resolve()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(root: Path) -> None:
    errors = []
    for name, expected in DATA_FILES.items():
        path = root / name
        if not path.is_file():
            errors.append(f"missing {name}")
            continue
        actual = md5(path)
        if actual != expected:
            errors.append(f"checksum mismatch for {name}: {actual}")
    if errors:
        raise RuntimeError(
            "Fashion-MNIST data is not prepared: "
            + "; ".join(errors)
            + ". Run `python -m experiments.c0c3_factorial.fashion_mnist prepare`."
        )


def prepare_dataset(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in DATA_FILES.items():
        destination = root / name
        if destination.is_file() and md5(destination) == expected:
            continue
        temporary = root / f".{name}.partial-{os.getpid()}"
        try:
            urllib.request.urlretrieve(f"{DOWNLOAD_BASE}/{name}", temporary)
            actual = md5(temporary)
            if actual != expected:
                raise RuntimeError(f"downloaded checksum mismatch for {name}: {actual}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    verify_dataset(root)
    manifest = {
        "schema_version": "1.0",
        "dataset": "Fashion-MNIST",
        "files_md5": DATA_FILES,
        "split": {
            "source": "official 60,000-example training set",
            "seed": SPLIT_SEED,
            "candidate_train_examples": TRAIN_EXAMPLES,
            "public_validation_examples": VALIDATION_EXAMPLES,
        },
        "sealed_test_examples": 10_000,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _read_idx(path: Path) -> tuple[tuple[int, ...], bytes]:
    with gzip.open(path, "rb") as handle:
        magic_bytes = handle.read(4)
        if len(magic_bytes) != 4:
            raise ValueError(f"truncated IDX header: {path.name}")
        zero_a, zero_b, data_type, dimensions = magic_bytes
        if (zero_a, zero_b, data_type) != (0, 0, 8) or dimensions not in {1, 3}:
            raise ValueError(f"unsupported IDX header in {path.name}")
        shape_bytes = handle.read(4 * dimensions)
        if len(shape_bytes) != 4 * dimensions:
            raise ValueError(f"truncated IDX shape: {path.name}")
        shape = struct.unpack(">" + "I" * dimensions, shape_bytes)
        payload = handle.read()
    expected = math.prod(shape)
    if len(payload) != expected:
        raise ValueError(
            f"IDX payload size mismatch in {path.name}: {len(payload)} != {expected}"
        )
    return tuple(int(value) for value in shape), payload


def load_dataset(root: Path):
    import torch

    verify_dataset(root)
    train_image_shape, train_image_bytes = _read_idx(
        root / "train-images-idx3-ubyte.gz"
    )
    train_label_shape, train_label_bytes = _read_idx(
        root / "train-labels-idx1-ubyte.gz"
    )
    test_image_shape, test_image_bytes = _read_idx(root / "t10k-images-idx3-ubyte.gz")
    test_label_shape, test_label_bytes = _read_idx(root / "t10k-labels-idx1-ubyte.gz")
    if train_image_shape != (60_000, 28, 28) or train_label_shape != (60_000,):
        raise ValueError("official training IDX shapes are not 60000x28x28 and 60000")
    if test_image_shape != (10_000, 28, 28) or test_label_shape != (10_000,):
        raise ValueError("official test IDX shapes are not 10000x28x28 and 10000")
    train_images = torch.frombuffer(bytearray(train_image_bytes), dtype=torch.uint8)
    train_images = train_images.reshape(train_image_shape)
    train_labels = torch.frombuffer(bytearray(train_label_bytes), dtype=torch.uint8)
    test_images = torch.frombuffer(bytearray(test_image_bytes), dtype=torch.uint8)
    test_images = test_images.reshape(test_image_shape)
    test_labels = torch.frombuffer(bytearray(test_label_bytes), dtype=torch.uint8)
    return train_images, train_labels.long(), test_images, test_labels.long()


def frozen_split_indices():
    import torch

    generator = torch.Generator(device="cpu").manual_seed(SPLIT_SEED)
    permutation = torch.randperm(60_000, generator=generator)
    return permutation[:TRAIN_EXAMPLES], permutation[TRAIN_EXAMPLES:]


def validation_score(correct: int, cross_entropy: float) -> float:
    """Encode correct-count-first, lower-cross-entropy-second ordering."""

    if isinstance(correct, bool) or not isinstance(correct, int) or correct < 0:
        raise ValueError("correct must be a nonnegative integer")
    if not math.isfinite(cross_entropy) or cross_entropy < 0:
        raise ValueError("cross_entropy must be finite and nonnegative")
    return float(correct) + 0.5 / (1.0 + cross_entropy)


def planned_optimizer_steps(training_examples: int, batch_size: int) -> int:
    """Count protected optimizer calls, including each pass's final short batch."""

    if training_examples < 1 or batch_size < 1:
        raise ValueError("training_examples and batch_size must be positive")
    full_passes, partial_pass = divmod(training_examples, TRAIN_EXAMPLES)
    steps = full_passes * math.ceil(TRAIN_EXAMPLES / batch_size)
    return steps + (math.ceil(partial_pass / batch_size) if partial_pass else 0)


def preflight_candidate_source(workspace: Path) -> str | None:
    """Reject broken interface submissions before consuming evaluator compute."""

    path = workspace / "train.py"
    if not path.is_file() or path.is_symlink():
        return "train.py is missing or unsafe"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename="train.py")
    except (OSError, SyntaxError, UnicodeError) as error:
        return f"train.py does not compile: {error}"
    required = {
        "build_model",
        "build_optimizer",
        "prepare_training_batch",
        "training_loss",
        "after_optimizer_step",
    }
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    missing = sorted(required - defined)
    if missing:
        return f"train.py is missing required functions: {', '.join(missing)}"

    def dotted_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = {alias.name.split(".", 1)[0] for alias in node.names}
            forbidden = sorted(imported & FORBIDDEN_CANDIDATE_IMPORTS)
            if forbidden:
                return f"train.py imports protected I/O module {forbidden[0]}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in FORBIDDEN_CANDIDATE_IMPORTS:
                return f"train.py imports protected I/O module {root}"
        elif isinstance(node, ast.Call):
            name = dotted_name(node.func).lower()
            leaf = name.rsplit(".", 1)[-1]
            if (
                leaf in FORBIDDEN_CANDIDATE_CALLS
                or name in FORBIDDEN_CANDIDATE_DOTTED_CALLS
            ):
                return f"train.py calls protected I/O operation {name}"
    return None


def _literal_assignment(path: Path, symbol: str) -> object | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return None
    result: object | None = None
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is None or not any(
            isinstance(target, ast.Name) and target.id == symbol for target in targets
        ):
            continue
        try:
            result = ast.literal_eval(value)
        except (ValueError, TypeError):
            return None
    return result


def _resolved_fidelity_policy(
    workspace: Path, args: argparse.Namespace
) -> tuple[list[int], list[float | None], dict[str, Any]]:
    terminal = int(args.training_examples)
    defaults = [
        int(value)
        for value in getattr(
            args, "ladder_levels", [25_000, 50_000, terminal]
        )
    ]
    default_thresholds = [
        None if value is None else float(value)
        for value in getattr(
            args, "promotion_thresholds", [0.82, 0.87, None]
        )
    ]
    if terminal not in defaults:
        defaults.append(terminal)
        default_thresholds.append(None)
    defaults = sorted(set(defaults))
    if len(default_thresholds) != len(defaults):
        raise ValueError("default ladder levels and thresholds must have equal length")
    if defaults[-1] != terminal:
        raise ValueError("the final default ladder rung must equal training-examples")
    receipt: dict[str, Any] = {
        "source": "controller_default",
        "accepted": False,
        "reason": "candidate policy absent",
        "levels": defaults,
        "promotion_thresholds": default_thresholds,
    }
    path = workspace / "train.py"
    raw_levels = _literal_assignment(path, "EVALUATION_LADDER")
    raw_thresholds = _literal_assignment(path, "EVALUATION_PROMOTION_THRESHOLDS")
    if not isinstance(raw_levels, list | tuple) or not isinstance(
        raw_thresholds, list | tuple
    ):
        return defaults, default_thresholds, receipt
    try:
        levels = [int(value) for value in raw_levels]
        thresholds = [
            None if value is None else float(value) for value in raw_thresholds
        ]
    except (TypeError, ValueError):
        receipt["reason"] = "candidate policy is not a literal numeric sequence"
        return defaults, default_thresholds, receipt
    if terminal not in levels:
        levels.append(terminal)
        if len(thresholds) == len(levels) - 1:
            thresholds.append(None)
    if len(thresholds) == len(levels) - 1:
        thresholds.append(None)
    if levels != sorted(set(levels)):
        receipt["reason"] = "candidate ladder levels must be sorted and unique"
        return defaults, default_thresholds, receipt
    minimum = int(getattr(args, "ladder_minimum", 10_000))
    maximum_rungs = int(getattr(args, "ladder_max_rungs", 6))
    if (
        not levels
        or levels[-1] != terminal
        or any(level < minimum or level > terminal for level in levels)
        or len(levels) > maximum_rungs
        or len(thresholds) != len(levels)
        or any(
            threshold is not None and not 0.0 <= threshold <= 1.0
            for threshold in thresholds
        )
    ):
        receipt["reason"] = "candidate policy exceeded evaluator-owned bounds"
        return defaults, default_thresholds, receipt
    thresholds[-1] = None
    receipt.update(
        {
            "source": "train.py",
            "accepted": True,
            "reason": "safe literal candidate policy accepted",
            "levels": levels,
            "promotion_thresholds": thresholds,
            "enforced_minimum_level": minimum,
            "enforced_terminal_level": terminal,
            "enforced_max_rungs": maximum_rungs,
        }
    )
    return levels, thresholds, receipt


def _load_program(workspace: Path):
    path = workspace / "train.py"
    if not path.is_file() or path.is_symlink():
        raise ValueError("train.py is missing or unsafe")
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    module_name = f"image_classifier_{source_hash}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _device(name: str):
    import torch

    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "the frozen Fashion-MNIST task requires an available MPS device"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA device is unavailable")
    return torch.device(name)


def _normalized(images, *, device):
    import torch

    return (
        images.to(device=device, dtype=torch.float32)
        .unsqueeze(1)
        .div_(255.0)
        .sub_(NORMALIZATION_MEAN)
        .div_(NORMALIZATION_STD)
    )


def _synchronize(device) -> None:
    import torch

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ModuleNotFoundError:
        pass
    import torch

    torch.manual_seed(seed)


def _score_model(
    model,
    *,
    evaluation_images,
    evaluation_labels,
    evaluation_batch_size: int,
    device,
) -> dict[str, float | int]:
    import torch
    from torch.nn import functional as F

    model.eval()
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for offset in range(0, len(evaluation_labels), evaluation_batch_size):
            images = _normalized(
                evaluation_images[offset : offset + evaluation_batch_size],
                device=device,
            )
            labels = evaluation_labels[
                offset : offset + evaluation_batch_size
            ].to(device=device, dtype=torch.long)
            logits = model(images)
            if logits.shape != (labels.shape[0], 10):
                raise ValueError(
                    "model must return one 10-class logit vector per image"
                )
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
            correct += int((logits.argmax(dim=1) == labels).sum())
    cases = int(len(evaluation_labels))
    cross_entropy = loss_sum / cases
    return {
        "validation_score": validation_score(correct, cross_entropy),
        "validation_correct": correct,
        "validation_accuracy": correct / cases,
        "validation_cross_entropy": cross_entropy,
        "evaluation_cases": cases,
    }


def evaluate(args: argparse.Namespace) -> int:
    import torch

    source_error = preflight_candidate_source(args.workspace.resolve())
    if source_error is not None:
        print(f"MODEL_CONTRACT_VIOLATION: {source_error}")
        return 3
    if args.training_examples < 1:
        raise ValueError("training-examples must be positive")
    if args.max_parameters < 1:
        raise ValueError("max-parameters must be positive")
    root = data_root(args.repo_root.resolve(), args.data_root)
    train_images, train_labels, test_images, test_labels = load_dataset(root)
    candidate_indices, validation_indices = frozen_split_indices()
    if args.layer == "A":
        evaluation_images = train_images[validation_indices]
        evaluation_labels = train_labels[validation_indices]
        split_name = "public_validation"
    else:
        evaluation_images = test_images
        evaluation_labels = test_labels
        split_name = "sealed_test"

    seed = int(os.environ.get("C0C3_RUN_SEED", args.seed)) % (2**63 - 1)
    _seed_everything(seed)
    device = _device(args.device)
    program = _load_program(args.workspace.resolve())
    batch_size = getattr(program, "BATCH_SIZE", 0)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError("BATCH_SIZE must be an integer")
    if not 16 <= batch_size <= 512:
        raise ValueError("BATCH_SIZE must be between 16 and 512")
    model = program.build_model().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters <= 0 or parameters > args.max_parameters:
        raise ValueError(
            f"model parameters must be in 1..{args.max_parameters}, got {parameters}"
        )
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not initial_parameters:
        raise ValueError("model must have trainable parameters")
    multi_fidelity = bool(getattr(args, "multi_fidelity", False))
    if multi_fidelity:
        ladder, promotion_thresholds, ladder_receipt = _resolved_fidelity_policy(
            args.workspace.resolve(), args
        )
    else:
        ladder = [int(args.training_examples)]
        promotion_thresholds = [None]
        ladder_receipt = {
            "source": "single_full_fidelity_evaluation",
            "accepted": False,
            "levels": ladder,
            "promotion_thresholds": promotion_thresholds,
        }
    terminal_exposure = ladder[-1]
    total_steps = planned_optimizer_steps(terminal_exposure, batch_size)
    optimizer = program.build_optimizer(model, total_steps)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("build_optimizer must return a torch optimizer")

    order_generator = torch.Generator(device="cpu").manual_seed(seed)
    seen = 0
    optimizer_steps = 0
    epoch_order = None
    epoch_offset = 0
    ladder_index = 0
    fidelity_stages: list[dict[str, Any]] = []
    latest_metrics: dict[str, float | int] | None = None
    screen_failure = False
    model.train()
    _synchronize(device)
    training_started = time.monotonic()
    while seen < terminal_exposure and not screen_failure:
        if epoch_order is None or epoch_offset >= TRAIN_EXAMPLES:
            epoch_order = candidate_indices[
                torch.randperm(TRAIN_EXAMPLES, generator=order_generator)
            ]
            epoch_offset = 0
        take = min(
            batch_size,
            terminal_exposure - seen,
            TRAIN_EXAMPLES - epoch_offset,
        )
        indices = epoch_order[epoch_offset : epoch_offset + take]
        epoch_offset += take
        images = _normalized(train_images[indices], device=device)
        labels = train_labels[indices].to(device=device, dtype=torch.long)
        step = optimizer_steps + 1
        images, labels = program.prepare_training_batch(
            images, labels, step, total_steps
        )
        if images.shape[0] != take or labels.shape[0] != take:
            raise ValueError("prepare_training_batch must preserve batch length")
        optimizer.zero_grad(set_to_none=True)
        loss = program.training_loss(model, images, labels, step, total_steps)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise TypeError("training_loss must return one scalar tensor")
        if not torch.isfinite(loss):
            raise ValueError("training_loss became non-finite")
        loss.backward()
        clip = getattr(program, "GRAD_CLIP_NORM", None)
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        optimizer.step()
        optimizer_steps += 1
        seen += take
        program.after_optimizer_step(optimizer, optimizer_steps, total_steps)
        while ladder_index < len(ladder) and seen >= ladder[ladder_index]:
            _synchronize(device)
            latest_metrics = _score_model(
                model,
                evaluation_images=evaluation_images,
                evaluation_labels=evaluation_labels,
                evaluation_batch_size=args.evaluation_batch_size,
                device=device,
            )
            threshold = promotion_thresholds[ladder_index]
            promoted = (
                ladder_index == len(ladder) - 1
                or threshold is None
                or float(latest_metrics["validation_accuracy"]) >= threshold
            )
            fidelity_stages.append(
                {
                    "stage": ladder_index + 1,
                    "requested_level": ladder[ladder_index],
                    "actual_examples_processed": seen,
                    "promotion_threshold": threshold,
                    "promoted": promoted,
                    # Keep the stage snapshot independent from ``latest_metrics``.
                    # The latter receives the full ``fidelity_stages`` list below;
                    # retaining this same dictionary here would create a circular
                    # object graph that cannot be serialized to JSON.
                    "metrics": dict(latest_metrics),
                }
            )
            ladder_index += 1
            if not promoted:
                screen_failure = True
                break
            model.train()
    _synchronize(device)
    training_seconds = time.monotonic() - training_started
    learned_change = any(
        not torch.equal(initial_parameters[name], parameter.detach().cpu())
        for name, parameter in model.named_parameters()
        if name in initial_parameters
    )
    if not learned_change:
        raise ValueError("training did not change any learned parameter")

    if latest_metrics is None:
        latest_metrics = _score_model(
            model,
            evaluation_images=evaluation_images,
            evaluation_labels=evaluation_labels,
            evaluation_batch_size=args.evaluation_batch_size,
            device=device,
        )
    reached_full = not screen_failure and seen == terminal_exposure
    latest_metrics.update(
        {
            "parameters": parameters,
            "examples_processed": seen,
            "optimizer_steps": optimizer_steps,
            "training_seconds": training_seconds,
            "batch_size": batch_size,
            "evaluation_split": split_name,
            "fidelity_highest_level": ladder[ladder_index - 1],
            "fidelity_reached_full": reached_full,
            "fidelity_stage_count": len(fidelity_stages),
            "fidelity_stages": fidelity_stages,
            "fidelity_policy": ladder_receipt,
        }
    )
    payload = {
        "schema_version": "1.0",
        "layer": args.layer,
        "valid": not screen_failure,
        "failure_kind": (
            "fidelity_screen_not_promoted" if screen_failure else None
        ),
        "metrics": latest_metrics,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"validation_score: {latest_metrics['validation_score']:.9f}")
    print(f"validation_accuracy: {latest_metrics['validation_accuracy']:.6f}")
    print(
        "validation_cross_entropy: "
        f"{latest_metrics['validation_cross_entropy']:.9f}"
    )
    print(f"parameters: {parameters}")
    print(f"examples_processed: {seen}")
    print(f"optimizer_steps: {optimizer_steps}")
    print(f"training_seconds: {training_seconds:.6f}")
    if multi_fidelity:
        print(f"fidelity_reached_full: {str(reached_full).lower()}")
        print(f"fidelity_stage_count: {len(fidelity_stages)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="download and verify the dataset")
    prepare.add_argument("--repo-root", type=Path, default=Path.cwd())
    prepare.add_argument("--data-root", type=Path)

    for name, layer, multi_fidelity in (
        ("evaluate", "A", False),
        ("evaluate-ladder", "A", True),
        ("holdout", "C", False),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--workspace", type=Path, required=True)
        command.add_argument("--repo-root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--data-root", type=Path)
        command.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
        command.add_argument(
            "--training-examples", type=int, default=DEFAULT_TRAINING_EXPOSURE
        )
        command.add_argument(
            "--max-parameters", type=int, default=DEFAULT_MAX_PARAMETERS
        )
        command.add_argument("--evaluation-batch-size", type=int, default=512)
        command.add_argument("--seed", type=int, default=20_260_823)
        if multi_fidelity:
            command.add_argument(
                "--ladder-levels",
                type=int,
                nargs="+",
                default=[25_000, 50_000, 100_000],
            )
            command.add_argument(
                "--promotion-thresholds",
                type=lambda value: None if value.casefold() == "none" else float(value),
                nargs="+",
                default=[0.82, 0.87, None],
            )
            command.add_argument("--ladder-minimum", type=int, default=10_000)
            command.add_argument("--ladder-max-rungs", type=int, default=6)
        command.set_defaults(
            handler=evaluate, layer=layer, multi_fidelity=multi_fidelity
        )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        root = data_root(args.repo_root.resolve(), args.data_root)
        print(json.dumps(prepare_dataset(root), indent=2, sort_keys=True))
        return 0
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
