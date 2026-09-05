"""Protected UCI HAR training, person-disjoint splits and executed affine MACs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

ARCHIVE_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
VALIDATION_SUBJECTS = (5, 14, 19, 26)
MAX_MACS = 2_000_000
SPLIT_SHA256 = {
    "train": "0ac73ffa96428be3a65689a0a45806100a85f492d308cbd76c01edfc83ecb9fc",
    "validation": "4e20ec110856b606d9597f970d4301a7480c94e62beaf1bf82839877c9438bc8",
    "holdout": "8cb18cad7ccf5e5c03fb2f36ff7ade68d235e38ccd9bedb74ac347db415dbc17",
}
CHANNELS = tuple(
    f"{signal}_{axis}"
    for signal in ("body_acc", "body_gyro", "total_acc")
    for axis in "xyz"
)
COUNTED_FORWARDS = {
    kind: kind.forward
    for kind in (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.RNN,
        nn.GRU,
        nn.LSTM,
        nn.RNNCell,
        nn.GRUCell,
        nn.LSTMCell,
    )
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_root(repo_root):
    return Path(
        os.environ.get(
            "RL4RL_UCI_HAR_DATA_ROOT", str(Path(repo_root) / "data/raw/uci-har")
        )
    )


def prepare_data(archive_path, destination):
    """Read only named members; never extract archive paths onto the filesystem."""
    if sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("UCI archive checksum mismatch")
    with zipfile.ZipFile(archive_path) as outer:
        nested = [n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip")]
        archive = (
            zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) if nested else outer
        )
        arrays = {}
        for split in ("train", "test"):
            prefix = "UCI HAR Dataset/" + split + "/"
            x = np.stack(
                [
                    np.loadtxt(
                        io.BytesIO(
                            archive.read(
                                prefix
                                + "Inertial Signals/"
                                + channel
                                + "_"
                                + split
                                + ".txt"
                            )
                        ),
                        dtype=np.float32,
                    )
                    for channel in CHANNELS
                ],
                axis=-1,
            )
            y = (
                np.loadtxt(
                    io.BytesIO(archive.read(prefix + "y_" + split + ".txt")),
                    dtype=np.int64,
                )
                - 1
            )
            subjects = np.loadtxt(
                io.BytesIO(archive.read(prefix + "subject_" + split + ".txt")),
                dtype=np.int64,
            )
            arrays[split] = (x, y, subjects)
        x, y, subjects = arrays["train"]
        mask = np.isin(subjects, VALIDATION_SUBJECTS)
        train_subjects = set(subjects[~mask].tolist())
        val_subjects = set(subjects[mask].tolist())
        test_subjects = set(arrays["test"][2].tolist())
        if (
            val_subjects != set(VALIDATION_SUBJECTS)
            or train_subjects & test_subjects
            or val_subjects & test_subjects
        ):
            raise ValueError("invalid subject split")
        mean = x[~mask].mean(axis=(0, 1), keepdims=True)
        std = x[~mask].std(axis=(0, 1), keepdims=True).clip(1e-6)
        parts = {
            "train": (x[~mask], y[~mask], subjects[~mask]),
            "validation": (x[mask], y[mask], subjects[mask]),
            "holdout": arrays["test"],
        }
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {
            "archive_sha256": ARCHIVE_SHA256,
            "channels": CHANNELS,
            "normalization": "per-channel training-subject mean and std",
            "splits": {},
        }
        for name, (features, labels, people) in parts.items():
            path = destination / (name + ".npz")
            np.savez(
                path, features=(features - mean) / std, labels=labels, subjects=people
            )
            manifest["splits"][name] = {
                "subjects": sorted(set(people.tolist())),
                "examples": len(labels),
                "sha256": sha256(path),
            }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if archive is not outer:
            archive.close()
    return manifest


def preflight_candidate_source(workspace):
    try:
        tree = ast.parse((Path(workspace) / "train.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [n.name.split(".")[0] for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]]
            else:
                roots = []
            if any(
                root not in {"__future__", "torch", "math", "typing"} for root in roots
            ):
                return "Only torch, math and typing imports are permitted."
            if isinstance(node, ast.Call):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else getattr(node.func, "attr", "")
                )
                if name in {
                    "open",
                    "eval",
                    "exec",
                    "compile",
                    "__import__",
                    "load",
                    "save",
                    "load_state_dict",
                    "set_num_threads",
                    "set_num_interop_threads",
                }:
                    return f"Forbidden candidate operation: {name}"
            if isinstance(node, ast.Attribute) and node.attr in {
                "__dict__",
                "__globals__",
                "__subclasses__",
            }:
                return "Runtime introspection is not permitted."
    except (OSError, SyntaxError) as error:
        return str(error)
    return None


def load_program(workspace):
    problem = preflight_candidate_source(workspace)
    if problem:
        raise ValueError(problem)
    spec = importlib.util.spec_from_file_location(
        "har_candidate", Path(workspace) / "train.py"
    )
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return program


class MacCounter(TorchDispatchMode):
    """Count executed affine MACs, including both recurrent directions.

    Biases, normalization, pooling, activations and elementwise gate products
    are excluded. Unsupported affine operations fail closed, rather than get
    assigned zero cost. Module subclasses cannot override a counted primitive.
    """

    def __init__(self, model):
        super().__init__()
        self.macs = 0
        self.depth = 0
        self.handles = []
        supported = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.RNN,
            nn.GRU,
            nn.LSTM,
            nn.RNNCell,
            nn.GRUCell,
            nn.LSTMCell,
        )
        harmless = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
        for module in model.modules():
            if (
                tuple(module.parameters(recurse=False))
                and type(module) not in supported + harmless
            ):
                raise ValueError(
                    f"No MAC rule for parameter-bearing {type(module).__name__}"
                )
            if isinstance(module, nn.LSTM) and module.proj_size:
                raise ValueError("Projected LSTM is not supported by the MAC counter")
            if type(module) in supported:
                if (
                    type(module).forward is not COUNTED_FORWARDS[type(module)]
                    or "forward" in module.__dict__
                    or module._forward_hooks
                    or module._forward_pre_hooks
                ):
                    raise ValueError(
                        "Counted primitives cannot override forward or install hooks"
                    )
                self.handles.append(module.register_forward_pre_hook(self._enter))
                self.handles.append(
                    module.register_forward_hook(self._exit, always_call=True)
                )

    def _enter(self, module, inputs):
        self.depth += 1

    def _exit(self, module, inputs, output):
        self.depth -= 1
        if output is None:
            return
        x = inputs[0]
        if isinstance(module, nn.Linear):
            self.macs += output.numel() * module.in_features
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            self.macs += (
                output.numel()
                * (module.in_channels // module.groups)
                * math.prod(module.kernel_size)
            )
        elif isinstance(module, (nn.RNNCell, nn.GRUCell, nn.LSTMCell)):
            gates = (
                4
                if isinstance(module, nn.LSTMCell)
                else 3
                if isinstance(module, nn.GRUCell)
                else 1
            )
            batch = x.numel() // module.input_size
            self.macs += (
                batch
                * gates
                * module.hidden_size
                * (module.input_size + module.hidden_size)
            )
        else:
            if not isinstance(x, torch.Tensor) or x.ndim != 3:
                raise ValueError("Recurrent inputs must be dense, batched sequences")
            batch, steps = (
                (x.shape[0], x.shape[1])
                if module.batch_first
                else (x.shape[1], x.shape[0])
            )
            directions = 2 if module.bidirectional else 1
            gates = (
                4
                if isinstance(module, nn.LSTM)
                else 3
                if isinstance(module, nn.GRU)
                else 1
            )
            for layer in range(module.num_layers):
                width = (
                    module.input_size if layer == 0 else module.hidden_size * directions
                )
                self.macs += (
                    batch
                    * steps
                    * directions
                    * gates
                    * module.hidden_size
                    * (width + module.hidden_size)
                )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        affine = (
            "mm.",
            "bmm.",
            "addmm.",
            "addbmm.",
            "baddbmm.",
            "convolution",
            "linear.",
            "lstm",
            "gru",
            "rnn",
            "scaled_dot_product",
            "embedding",
        )
        if self.depth == 0 and any(token in name for token in affine):
            raise ValueError(f"Use counted nn primitives for affine inference: {name}")
        return func(*args, **(kwargs or {}))

    def close(self):
        for handle in self.handles:
            handle.remove()


def read_split(root, split, manifest):
    path = root / (split + ".npz")
    expected = SPLIT_SHA256[split]
    if manifest["splits"][split]["sha256"] != expected or sha256(path) != expected:
        raise ValueError("data cache checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        return torch.from_numpy(data["features"].copy()), torch.from_numpy(
            data["labels"].copy()
        )


def score(model, features, labels, batch_size=128):
    model.eval()
    confusion = torch.zeros(6, 6, dtype=torch.int64)
    loss = 0.0
    counter = MacCounter(model)
    try:
        with torch.no_grad(), counter:
            for offset in range(0, len(labels), batch_size):
                y = labels[offset : offset + batch_size]
                logits = model(features[offset : offset + batch_size])
                if logits.shape != (len(y), 6) or not torch.isfinite(logits).all():
                    raise ValueError("model must return finite [batch,6] logits")
                loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
                predictions = logits.argmax(dim=-1)
                confusion += torch.bincount(6 * y + predictions, minlength=36).reshape(
                    6, 6
                )
    finally:
        counter.close()
    correct = int(confusion.diag().sum())
    f1 = (
        2 * confusion.diag() / (confusion.sum(0) + confusion.sum(1)).clamp_min(1)
    ).mean()
    cost = counter.macs / len(labels)
    if not 0 < cost <= MAX_MACS:
        raise ValueError(
            "inference MACs must be positive and at most 2,000,000 per example"
        )
    return {
        "validation_accuracy": correct / len(labels),
        "validation_correct": correct,
        "validation_examples": len(labels),
        "validation_macro_f1": float(f1),
        "validation_cross_entropy": loss / len(labels),
        "inference_macs": cost,
        "total_inference_macs": counter.macs,
    }


def evaluate(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    # Avoid distinct fused RNN paths; CPU/OS roundoff can still differ.
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    seed = int(os.environ.get("C0C3_RUN_SEED", "20260905"))
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    root = data_root(args.repo_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("unexpected dataset manifest")
    train_x, train_y = read_split(root, "train", manifest)
    program = load_program(args.workspace)
    source_hash = sha256(Path(args.workspace) / "train.py")
    model = program.build_model().cpu()
    parameters = sum(p.numel() for p in model.parameters())
    if not 0 < parameters <= args.max_parameters:
        raise ValueError("model exceeds parameter limit")
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    batch_size = program.BATCH_SIZE
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 16 <= batch_size <= 512
    ):
        raise ValueError("BATCH_SIZE must be an integer from 16 to 512")
    # No labels used for this pre-training complexity check.
    score(model, train_x[:8], train_y[:8], batch_size=8)
    passes, remainder = divmod(args.training_examples, len(train_y))
    total_steps = passes * math.ceil(len(train_y) / batch_size) + math.ceil(
        remainder / batch_size
    )
    optimizer = program.build_optimizer(model, total_steps)
    started = time.monotonic()
    seen, steps, offset = 0, 0, len(train_y)
    order = None
    generator = torch.Generator().manual_seed(seed)
    while seen < args.training_examples:
        if offset == len(train_y):
            order = torch.randperm(len(train_y), generator=generator)
            offset = 0
        take = min(batch_size, args.training_examples - seen, len(train_y) - offset)
        indices = order[offset : offset + take]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = program.training_loss(
            model,
            train_x[indices].clone(),
            train_y[indices].clone(),
            steps + 1,
            total_steps,
        )
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("training loss must be a finite scalar")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(program.GRAD_CLIP_NORM))
        optimizer.step()
        steps += 1
        seen += take
        offset += take
        program.after_optimizer_step(optimizer, steps, total_steps)
    training_seconds = time.monotonic() - started
    if not any(
        not torch.equal(original[n], p.detach()) for n, p in model.named_parameters()
    ):
        raise ValueError("training did not change model parameters")
    if sha256(Path(args.workspace) / "train.py") != source_hash:
        raise ValueError("source changed during training")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "parameters": parameters,
        "training_steps": steps,
        "training_examples": seen,
        "seed": seed,
    }
    checkpoints = Path(args.workspace) / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    torch.save(checkpoint, checkpoints / "best.pt")
    torch.save(checkpoint, checkpoints / "last.pt")
    verified = program.build_model().cpu()
    verified.load_state_dict(
        torch.load(checkpoints / "best.pt", weights_only=True)["model_state_dict"]
    )
    if sum(p.numel() for p in verified.parameters()) != parameters:
        raise ValueError("model architecture changed after training")
    # The final test split is never opened by normal candidate evaluation.
    evaluation_split = "holdout" if args.holdout else "validation"
    eval_x, eval_y = read_split(root, evaluation_split, manifest)
    metrics = score(verified, eval_x, eval_y)
    metrics.update(
        parameters=parameters,
        examples_processed=seen,
        optimizer_steps=steps,
        training_seconds=training_seconds,
        batch_size=batch_size,
        evaluation_seed=seed,
        dataset_sha256=SPLIT_SHA256[evaluation_split],
    )
    Path(args.output).write_text(
        json.dumps({"valid": True, "metrics": metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--training-examples", type=int, default=50_000)
    run.add_argument("--max-parameters", type=int, default=100_000)
    run.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-data":
        print(json.dumps(prepare_data(args.archive, args.output), indent=2))
    else:
        if args.training_examples < 1 or args.max_parameters < 1:
            parser.error("budgets must be positive")
        try:
            evaluate(args)
        except (ValueError, RuntimeError) as error:
            print(f"MODEL_CONTRACT_VIOLATION: {error}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
