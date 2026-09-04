#!/usr/bin/env python3
"""Protected Mini Speech Commands recurrent keyword-spotting evaluator."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import urllib.request
import wave
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DATA_ROOT_ENV = "RL4RL_TINY_KWS_DATA_ROOT"
DOWNLOAD_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "mini_speech_commands.zip"
)
ARCHIVE_SHA256 = "49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0"
LABELS = ("down", "go", "left", "no", "right", "stop", "up", "yes")
SAMPLE_RATE = 16_000
WAVEFORM_SAMPLES = 16_000
MEL_BANDS = 20
OUTPUT_FRAMES = 32
SPLIT_SEED = 20_260_903
DEFAULT_TRAINING_EXPOSURE = 50_000
DEFAULT_MAX_PARAMETERS = 100_000
MAX_RECURRENT_STEPS = 64
FEATURE_CACHE = "logmel-32x20.pt"
MANIFEST = "manifest.json"

FORBIDDEN_IMPORTS = frozenset(
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
        "torchaudio",
        "urllib",
        "wave",
        "zipfile",
    }
)
FORBIDDEN_CALLS = frozenset({"__import__", "compile", "eval", "exec", "open"})
FORBIDDEN_DOTTED_CALLS = frozenset(
    {
        "f.linear",
        "np.load",
        "numpy.load",
        "torch.bmm",
        "torch.einsum",
        "torch.hub.load",
        "torch.load",
        "torch.matmul",
        "torch.mm",
    }
)
FORBIDDEN_ATTRIBUTES = frozenset({"matmul"})


def data_root(repo_root: Path, configured: Path | None = None) -> Path:
    if configured is not None:
        return configured.expanduser().resolve()
    environment = os.environ.get(DATA_ROOT_ENV)
    if environment:
        return Path(environment).expanduser().resolve()
    return (repo_root / "data/raw/tiny-kws-rnn").resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _speaker_id(path: Path) -> str:
    return path.stem.split("_nohash_", 1)[0]


def _speaker_split(speaker: str) -> str:
    value = (
        int.from_bytes(
            hashlib.sha256(f"{SPLIT_SEED}:{speaker}".encode()).digest()[:8], "big"
        )
        % 100
    )
    if value < 80:
        return "train"
    if value < 90:
        return "validation"
    return "layer_c"


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        root = destination.resolve()
        for item in handle.infolist():
            if item.filename.startswith("__MACOSX/"):
                continue
            target = (destination / item.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError("unsafe path in Mini Speech Commands archive")
            handle.extract(item, destination)
    return destination / "mini_speech_commands"


def _read_wave(path: Path):
    import torch

    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != SAMPLE_RATE
        ):
            raise ValueError(f"unsupported WAV format: {path.name}")
        payload = handle.readframes(handle.getnframes())
    values = torch.frombuffer(bytearray(payload), dtype=torch.int16).float().div_(32768)
    if values.numel() < WAVEFORM_SAMPLES:
        values = torch.nn.functional.pad(values, (0, WAVEFORM_SAMPLES - values.numel()))
    return values[:WAVEFORM_SAMPLES]


def _mel_filter():
    import torch

    n_fft = 512
    frequencies = torch.linspace(0.0, SAMPLE_RATE / 2, n_fft // 2 + 1)

    def hz_to_mel(value):
        return 2595.0 * torch.log10(1.0 + value / 700.0)

    def mel_to_hz(value):
        return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)

    edges = mel_to_hz(
        torch.linspace(
            hz_to_mel(torch.tensor(40.0)),
            hz_to_mel(torch.tensor(7600.0)),
            MEL_BANDS + 2,
        )
    )
    filters = torch.zeros(MEL_BANDS, frequencies.numel())
    for index in range(MEL_BANDS):
        left, center, right = edges[index : index + 3]
        filters[index] = torch.minimum(
            (frequencies - left) / (center - left),
            (right - frequencies) / (right - center),
        ).clamp_min(0.0)
    return filters


def _logmel(path: Path, mel_filter, window):
    import torch

    waveform = _read_wave(path)
    spectrum = (
        torch.stft(
            waveform,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=window,
            center=True,
            return_complex=True,
        )
        .abs()
        .square()
    )
    mel = torch.log1p(mel_filter @ spectrum)
    mel = torch.nn.functional.interpolate(
        mel.unsqueeze(0), size=OUTPUT_FRAMES, mode="linear", align_corners=False
    ).squeeze(0)
    return mel.transpose(0, 1).contiguous()


def prepare_dataset(
    root: Path, *, archive_source: Path | None = None
) -> dict[str, Any]:
    import torch

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "mini_speech_commands.zip"
    if archive_source is not None:
        source = archive_source.expanduser().resolve()
        if sha256(source) != ARCHIVE_SHA256:
            raise RuntimeError(
                "provided Mini Speech Commands archive checksum mismatch"
            )
        if not archive.is_file() or sha256(archive) != ARCHIVE_SHA256:
            shutil.copyfile(source, archive)
    if not archive.is_file() or sha256(archive) != ARCHIVE_SHA256:
        partial = root / f".mini_speech_commands.zip.partial-{os.getpid()}"
        try:
            urllib.request.urlretrieve(DOWNLOAD_URL, partial)
            if sha256(partial) != ARCHIVE_SHA256:
                raise RuntimeError("downloaded Mini Speech Commands checksum mismatch")
            os.replace(partial, archive)
        finally:
            partial.unlink(missing_ok=True)

    extracted = root / "extracted"
    source_root = extracted / "mini_speech_commands"
    if not source_root.is_dir():
        _safe_extract(archive, extracted)
    files: list[tuple[Path, int, str, str]] = []
    for label_index, label in enumerate(LABELS):
        for path in sorted((source_root / label).glob("*.wav")):
            speaker = _speaker_id(path)
            files.append((path, label_index, speaker, _speaker_split(speaker)))
    if not files:
        raise RuntimeError(
            "Mini Speech Commands archive contains no command recordings"
        )

    mel_filter = _mel_filter()
    window = torch.hann_window(400)
    split_features: dict[str, list[Any]] = {
        name: [] for name in ("train", "validation", "layer_c")
    }
    split_labels: dict[str, list[int]] = {name: [] for name in split_features}
    split_speakers: dict[str, set[str]] = {name: set() for name in split_features}
    for index, (path, label, speaker, split) in enumerate(files, start=1):
        split_features[split].append(_logmel(path, mel_filter, window))
        split_labels[split].append(label)
        split_speakers[split].add(speaker)
        if index % 500 == 0:
            print(f"precomputed {index}/{len(files)} recordings", flush=True)
    if any(not values for values in split_features.values()):
        raise RuntimeError("speaker hash split produced an empty partition")
    training_stack = torch.stack(split_features["train"])
    mean = training_stack.mean(dim=(0, 1))
    std = training_stack.std(dim=(0, 1)).clamp_min(1e-6)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "labels": LABELS,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
    }
    for split in split_features:
        features = (torch.stack(split_features[split]) - mean) / std
        payload[f"{split}_features"] = features.to(torch.float16)
        payload[f"{split}_labels"] = torch.tensor(split_labels[split], dtype=torch.long)
    cache = root / FEATURE_CACHE
    torch.save(payload, cache)
    manifest = {
        "schema_version": "1.0",
        "dataset": "TensorFlow Mini Speech Commands",
        "archive_sha256": ARCHIVE_SHA256,
        "feature_cache_sha256": sha256(cache),
        "frontend": {
            "sample_rate": SAMPLE_RATE,
            "waveform_samples": WAVEFORM_SAMPLES,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "mel_bands": MEL_BANDS,
            "output_frames": OUTPUT_FRAMES,
        },
        "speaker_split_seed": SPLIT_SEED,
        "splits": {
            split: {
                "recordings": len(split_features[split]),
                "speakers": len(split_speakers[split]),
            }
            for split in split_features
        },
        "speaker_disjoint": not any(
            split_speakers[left] & split_speakers[right]
            for left, right in (
                ("train", "validation"),
                ("train", "layer_c"),
                ("validation", "layer_c"),
            )
        ),
    }
    (root / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_dataset(root)
    return manifest


def verify_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST
    cache = root / FEATURE_CACHE
    archive = root / "mini_speech_commands.zip"
    if not manifest_path.is_file() or not cache.is_file() or not archive.is_file():
        raise RuntimeError(
            "TinyKWS data is not prepared; run `python -m "
            "experiments.c0c3_factorial.tiny_kws_rnn prepare --repo-root .`"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(archive) != ARCHIVE_SHA256:
        raise RuntimeError("TinyKWS source archive checksum mismatch")
    if sha256(cache) != manifest.get("feature_cache_sha256"):
        raise RuntimeError("TinyKWS protected feature cache checksum mismatch")
    if manifest.get("speaker_disjoint") is not True:
        raise RuntimeError("TinyKWS speaker partitions are not disjoint")
    return manifest


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def preflight_candidate_source(workspace: Path) -> str | None:
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
    if missing := sorted(required - defined):
        return f"train.py is missing required functions: {', '.join(missing)}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = {alias.name.split(".", 1)[0] for alias in node.names}
            if forbidden := sorted(imported & FORBIDDEN_IMPORTS):
                return f"train.py imports protected I/O module {forbidden[0]}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in FORBIDDEN_IMPORTS:
                return f"train.py imports protected I/O module {root}"
        elif isinstance(node, ast.MatMult):
            return "train.py uses uncounted matrix multiplication"
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            return f"train.py uses uncounted operation {node.attr}"
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func).lower()
            leaf = name.rsplit(".", 1)[-1]
            if leaf in FORBIDDEN_CALLS or name in FORBIDDEN_DOTTED_CALLS:
                return f"train.py calls protected or uncounted operation {name}"
            if leaf == "parameter":
                return "train.py must create learned matrices through countable modules"
    return None


def _load_program(workspace: Path):
    path = workspace / "train.py"
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    module_name = f"tiny_kws_{source_hash}"
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
        raise RuntimeError("the frozen TinyKWS task requires an available MPS device")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA device is unavailable")
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ModuleNotFoundError:
        pass
    import torch

    torch.manual_seed(seed)


def _synchronize(device) -> None:
    import torch

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_tensors(state: Any) -> Iterable[Any]:
    import torch

    if isinstance(state, torch.Tensor):
        yield state
    elif isinstance(state, tuple | list):
        for value in state:
            yield from _state_tensors(value)
    else:
        raise TypeError("recurrent state must be a tensor or tuple/list of tensors")


def _state_index(state: Any, indices):
    import torch

    if isinstance(state, torch.Tensor):
        return state.index_select(0, indices)
    values = [_state_index(value, indices) for value in state]
    return tuple(values) if isinstance(state, tuple) else values


def _state_zeros(state: Any):
    import torch

    if isinstance(state, torch.Tensor):
        return torch.zeros_like(state)
    values = [_state_zeros(value) for value in state]
    return tuple(values) if isinstance(state, tuple) else values


def _state_difference(left: Any, right: Any) -> float:
    import torch

    values = [
        (a - b).detach().float().abs().mean()
        for a, b in zip(_state_tensors(left), _state_tensors(right), strict=True)
    ]
    return float(torch.stack(values).mean())


def _frame_schedule(model, available_frames: int) -> list[int]:
    raw = (
        model.frame_schedule(available_frames)
        if callable(getattr(model, "frame_schedule", None))
        else list(range(available_frames))
    )
    if not isinstance(raw, list | tuple):
        raise TypeError("frame_schedule must return a list or tuple")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise TypeError("frame_schedule entries must be integers")
    schedule = [int(value) for value in raw]
    if (
        len(schedule) < 2
        or len(schedule) > MAX_RECURRENT_STEPS
        or schedule != sorted(set(schedule))
        or schedule[0] < 0
        or schedule[-1] >= available_frames
    ):
        raise ValueError(
            "frame_schedule must contain 2..64 unique causal frame indices"
        )
    return schedule


class _MacCounter:
    def __init__(self, model) -> None:
        import torch

        self.macs = 0
        self.classifier_macs = 0
        self.handles = []
        supported = (
            torch.nn.Linear,
            torch.nn.RNNCell,
            torch.nn.GRUCell,
            torch.nn.LSTMCell,
            torch.nn.RNN,
            torch.nn.GRU,
            torch.nn.LSTM,
        )
        harmless = (
            torch.nn.Identity,
            torch.nn.Dropout,
            torch.nn.LayerNorm,
            torch.nn.BatchNorm1d,
            torch.nn.ReLU,
            torch.nn.GELU,
            torch.nn.SiLU,
            torch.nn.Tanh,
            torch.nn.Sigmoid,
        )
        for module in model.modules():
            children = tuple(module.children())
            direct_parameters = tuple(module.parameters(recurse=False))
            if direct_parameters and not isinstance(module, supported + harmless):
                raise ValueError(
                    "parameter-bearing module "
                    f"{type(module).__name__} has no exact MAC rule"
                )
            if isinstance(module, supported):
                if isinstance(module, torch.nn.RNN | torch.nn.GRU | torch.nn.LSTM):
                    if module.bidirectional:
                        raise ValueError("bidirectional recurrence is not causal")
                    if isinstance(module, torch.nn.LSTM) and module.proj_size:
                        raise ValueError(
                            "projected LSTM does not have an exact MAC rule"
                        )
                self.handles.append(module.register_forward_hook(self._hook))
            elif (
                not children and direct_parameters and not isinstance(module, harmless)
            ):
                raise ValueError(f"uncountable learned module {type(module).__name__}")

    def _hook(self, module, inputs, output) -> None:
        import torch

        del output
        data = inputs[0]
        rows = int(data.numel() // data.shape[-1])
        if isinstance(module, torch.nn.Linear):
            self.macs += rows * module.in_features * module.out_features
        elif isinstance(module, torch.nn.RNNCell):
            self.macs += rows * (
                module.input_size * module.hidden_size
                + module.hidden_size * module.hidden_size
            )
        elif isinstance(module, torch.nn.GRUCell):
            self.macs += (
                rows
                * 3
                * (
                    module.input_size * module.hidden_size
                    + module.hidden_size * module.hidden_size
                )
            )
        elif isinstance(module, torch.nn.LSTMCell):
            self.macs += (
                rows
                * 4
                * (
                    module.input_size * module.hidden_size
                    + module.hidden_size * module.hidden_size
                )
            )
        elif isinstance(module, torch.nn.RNN | torch.nn.GRU | torch.nn.LSTM):
            batch = int(data.shape[0] if module.batch_first else data.shape[1])
            timesteps = int(data.shape[1] if module.batch_first else data.shape[0])
            gates = 1 if isinstance(module, torch.nn.RNN) else 3
            if isinstance(module, torch.nn.LSTM):
                gates = 4
            for layer in range(module.num_layers):
                input_size = module.input_size if layer == 0 else module.hidden_size
                self.macs += (
                    batch
                    * timesteps
                    * gates
                    * (
                        input_size * module.hidden_size
                        + module.hidden_size * module.hidden_size
                    )
                )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _run_sequence(model, frames, *, counter: _MacCounter | None = None):
    import torch

    batch_size = int(frames.shape[0])
    schedule = _frame_schedule(model, int(frames.shape[1]))
    state = model.initial_state(batch_size, frames.device, frames.dtype)
    tensors = tuple(_state_tensors(state))
    if not tensors or any(tensor.shape[0] != batch_size for tensor in tensors):
        raise ValueError("initial_state must be batch-first")
    peak_hidden = sum(int(tensor.numel()) for tensor in tensors)
    recurrent_steps = 0
    exit_function = getattr(model, "exit_mask", None)
    dynamic = callable(exit_function)
    if not dynamic:
        sequence_function = getattr(model, "recurrent_sequence", None)
        if callable(sequence_function):
            state = sequence_function(frames[:, schedule, :], state)
            recurrent_steps = batch_size * len(schedule)
            if any(tensor.shape[0] != batch_size for tensor in _state_tensors(state)):
                raise ValueError("recurrent_sequence must preserve batch length")
            peak_hidden = max(
                peak_hidden,
                sum(int(tensor.numel()) for tensor in _state_tensors(state)),
            )
        else:
            for frame_index in schedule:
                state = model.recurrent_step(frames[:, frame_index, :], state)
                if any(
                    tensor.shape[0] != batch_size for tensor in _state_tensors(state)
                ):
                    raise ValueError("recurrent_step must preserve active batch length")
                recurrent_steps += batch_size
                peak_hidden = max(
                    peak_hidden,
                    sum(int(tensor.numel()) for tensor in _state_tensors(state)),
                )
        before_classifier = counter.macs if counter is not None else 0
        logits = model.classify(state)
        if counter is not None:
            counter.classifier_macs += counter.macs - before_classifier
        return logits, recurrent_steps, peak_hidden, [len(schedule)] * batch_size

    active = torch.arange(batch_size, device=frames.device)
    final_logits = torch.zeros(
        batch_size, len(LABELS), device=frames.device, dtype=frames.dtype
    )
    depths = torch.zeros(batch_size, dtype=torch.long, device=frames.device)
    for position, frame_index in enumerate(schedule):
        state = model.recurrent_step(frames[active, frame_index, :], state)
        active_count = int(active.numel())
        recurrent_steps += active_count
        peak_hidden = max(
            peak_hidden, sum(int(tensor.numel()) for tensor in _state_tensors(state))
        )
        before_classifier = counter.macs if counter is not None else 0
        logits = model.classify(state)
        if counter is not None:
            counter.classifier_macs += counter.macs - before_classifier
        if logits.shape != (active_count, len(LABELS)):
            raise ValueError(
                "classify must return one eight-class logit vector per state"
            )
        final_step = position == len(schedule) - 1
        if final_step:
            exiting = torch.ones(active_count, dtype=torch.bool, device=frames.device)
        elif position < 1:
            exiting = torch.zeros(active_count, dtype=torch.bool, device=frames.device)
        else:
            exiting = exit_function(state, logits, position + 1, len(schedule))
            if not isinstance(exiting, torch.Tensor) or exiting.shape != (
                active_count,
            ):
                raise ValueError("exit_mask must return one boolean per active example")
            exiting = exiting.to(device=frames.device, dtype=torch.bool)
        if bool(exiting.any()):
            final_logits = final_logits.index_copy(0, active[exiting], logits[exiting])
            depths = depths.index_fill(0, active[exiting], position + 1)
        keep = ~exiting
        if not bool(keep.any()):
            break
        active = active[keep]
        state = _state_index(state, torch.nonzero(keep, as_tuple=False).flatten())
    return final_logits, recurrent_steps, peak_hidden, depths.detach().cpu().tolist()


def _recurrent_contract(model, sample) -> dict[str, float]:
    import torch

    schedule = _frame_schedule(model, int(sample.shape[1]))
    state = model.initial_state(int(sample.shape[0]), sample.device, sample.dtype)
    first = model.recurrent_step(sample[:, schedule[0], :], state)
    second_from_first = model.recurrent_step(sample[:, schedule[1], :], first)
    second_from_zero = model.recurrent_step(
        sample[:, schedule[1], :], _state_zeros(first)
    )
    dependence = _state_difference(second_from_first, second_from_zero)
    state_change = _state_difference(first, state)
    logits = model.classify(second_from_first)
    ablated = model.classify(_state_zeros(second_from_first))
    logit_dependence = float((logits - ablated).detach().float().abs().mean())
    if state_change <= 1e-7:
        raise ValueError("recurrent state does not update")
    if dependence <= 1e-7:
        raise ValueError("next recurrent state does not depend on previous state")
    if logit_dependence <= 1e-6:
        raise ValueError("final logits do not materially depend on recurrent state")
    if logits.shape != (sample.shape[0], len(LABELS)):
        raise ValueError("classify must return one eight-class logit vector per state")
    if not torch.isfinite(logits).all():
        raise ValueError("model logits are not finite")
    sequence_function = getattr(model, "recurrent_sequence", None)
    if callable(sequence_function):
        iterative = model.initial_state(
            int(sample.shape[0]), sample.device, sample.dtype
        )
        for frame_index in schedule:
            iterative = model.recurrent_step(sample[:, frame_index, :], iterative)
        sequence = model.initial_state(
            int(sample.shape[0]), sample.device, sample.dtype
        )
        sequence = sequence_function(sample[:, schedule, :], sequence)
        if _state_difference(iterative, sequence) > 2e-5:
            raise ValueError(
                "recurrent_sequence is not equivalent to causal recurrent_step "
                "execution"
            )
    return {
        "state_update_effect": state_change,
        "state_dependence_effect": dependence,
        "recurrent_logit_effect": logit_dependence,
    }


def _score_model(model, features, labels, batch_size: int, device):
    import torch
    from torch.nn import functional as F

    model.eval()
    counter = _MacCounter(model)
    correct = 0
    loss_sum = 0.0
    total_steps = 0
    peak_hidden = 0
    exit_depths: list[int] = []
    try:
        with torch.no_grad():
            for offset in range(0, len(labels), batch_size):
                batch_features = features[offset : offset + batch_size].to(
                    device=device, dtype=torch.float32
                )
                batch_labels = labels[offset : offset + batch_size].to(device=device)
                logits, steps, peak, batch_depths = _run_sequence(
                    model, batch_features, counter=counter
                )
                if logits.shape != (batch_labels.shape[0], len(LABELS)):
                    raise ValueError("model must produce eight command logits")
                loss_sum += float(
                    F.cross_entropy(logits, batch_labels, reduction="sum")
                )
                correct += int((logits.argmax(1) == batch_labels).sum())
                total_steps += steps
                peak_hidden = max(peak_hidden, peak)
                exit_depths.extend(batch_depths)
    finally:
        counter.close()
    cases = int(len(labels))
    ordered = sorted(exit_depths)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    max_total_steps = cases * MAX_RECURRENT_STEPS
    inference_cost = (counter.macs * (max_total_steps + 1) + total_steps) * (
        DEFAULT_MAX_PARAMETERS + 1
    ) + parameters
    return {
        "validation_accuracy": correct / cases,
        "validation_cross_entropy": loss_sum / cases,
        "validation_correct": correct,
        "evaluation_cases": cases,
        "inference_cost": inference_cost,
        "total_inference_macs": counter.macs,
        "recurrent_macs": counter.macs - counter.classifier_macs,
        "recurrent_steps": total_steps,
        "mean_recurrent_steps": total_steps / cases,
        "median_recurrent_steps": ordered[cases // 2],
        "p95_recurrent_steps": ordered[min(cases - 1, math.ceil(0.95 * cases) - 1)],
        "maximum_recurrent_steps": max(ordered),
        "parameters": parameters,
        "peak_hidden_elements": peak_hidden,
    }


def evaluate(args: argparse.Namespace) -> int:
    import torch

    source_error = preflight_candidate_source(args.workspace.resolve())
    if source_error is not None:
        print(f"MODEL_CONTRACT_VIOLATION: {source_error}")
        return 3
    root = data_root(args.repo_root.resolve(), args.data_root)
    verify_dataset(root)
    dataset = torch.load(root / FEATURE_CACHE, map_location="cpu", weights_only=True)
    split = "validation" if args.layer == "A" else "layer_c"
    evaluation_features = dataset[f"{split}_features"]
    evaluation_labels = dataset[f"{split}_labels"]
    train_features = dataset["train_features"]
    train_labels = dataset["train_labels"]
    seed = int(os.environ.get("C0C3_RUN_SEED", args.seed)) % (2**63 - 1)
    _seed_everything(seed)
    device = _device(args.device)
    program = _load_program(args.workspace.resolve())
    batch_size = getattr(program, "BATCH_SIZE", 0)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 16 <= batch_size <= 512
    ):
        raise ValueError("BATCH_SIZE must be an integer between 16 and 512")
    model = program.build_model().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters <= 0 or parameters > args.max_parameters:
        raise ValueError(
            f"model parameters must be in 1..{args.max_parameters}, got {parameters}"
        )
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not initial:
        raise ValueError("model must have trainable parameters")
    sample = train_features[:4].to(device=device, dtype=torch.float32)
    contract_metrics = _recurrent_contract(model, sample)
    # Constructing the counter before training also rejects uncounted learned
    # modules even if a particular minibatch would not call them.
    counter = _MacCounter(model)
    counter.close()

    training_examples = int(args.training_examples)
    if not 25_000 <= training_examples <= 50_000:
        raise ValueError("training-examples must be between 25,000 and 50,000")
    steps = math.ceil(training_examples / batch_size)
    optimizer = program.build_optimizer(model, steps)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("build_optimizer must return a torch optimizer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    seen = 0
    optimizer_steps = 0
    order = torch.randperm(len(train_labels), generator=generator)
    offset = 0
    model.train()
    _synchronize(device)
    started = time.monotonic()
    while seen < training_examples:
        if offset >= len(order):
            order = torch.randperm(len(train_labels), generator=generator)
            offset = 0
        take = min(batch_size, training_examples - seen, len(order) - offset)
        indices = order[offset : offset + take]
        offset += take
        frames = train_features[indices].to(device=device, dtype=torch.float32)
        labels = train_labels[indices].to(device=device)
        frames, labels = program.prepare_training_batch(
            frames, labels, optimizer_steps + 1, steps
        )
        if frames.shape != (take, OUTPUT_FRAMES, MEL_BANDS) or labels.shape != (take,):
            raise ValueError(
                "prepare_training_batch must preserve the protected batch shape"
            )
        optimizer.zero_grad(set_to_none=True)
        logits, _, _, _ = _run_sequence(model, frames)
        loss = program.training_loss(model, logits, labels, optimizer_steps + 1, steps)
        if (
            not isinstance(loss, torch.Tensor)
            or loss.ndim != 0
            or not torch.isfinite(loss)
        ):
            raise ValueError("training_loss must return one finite scalar tensor")
        loss.backward()
        clip = getattr(program, "GRAD_CLIP_NORM", None)
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        optimizer.step()
        optimizer_steps += 1
        seen += take
        program.after_optimizer_step(optimizer, optimizer_steps, steps)
    _synchronize(device)
    training_seconds = time.monotonic() - started
    changed = {
        name
        for name, parameter in model.named_parameters()
        if name in initial and not torch.equal(initial[name], parameter.detach().cpu())
    }
    if not changed:
        raise ValueError("training did not change any learned parameter")
    post_contract = _recurrent_contract(model, sample)
    if not any("classifier" not in name.casefold() for name in changed):
        raise ValueError("training did not change any recurrent-path parameter")

    metrics = _score_model(
        model,
        evaluation_features,
        evaluation_labels,
        args.evaluation_batch_size,
        device,
    )
    metrics.update(contract_metrics)
    metrics.update({f"trained_{key}": value for key, value in post_contract.items()})
    metrics.update(
        {
            "examples_processed": seen,
            "optimizer_steps": optimizer_steps,
            "training_seconds": training_seconds,
            "batch_size": batch_size,
            "evaluation_split": "public_speaker_disjoint_validation"
            if args.layer == "A"
            else "sealed_speaker_disjoint_layer_c",
        }
    )
    payload = {
        "schema_version": "1.0",
        "layer": args.layer,
        "valid": True,
        "failure_kind": None,
        "metrics": metrics,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in (
        "validation_accuracy",
        "validation_cross_entropy",
        "inference_cost",
        "total_inference_macs",
        "recurrent_macs",
        "recurrent_steps",
        "parameters",
        "peak_hidden_elements",
        "examples_processed",
        "training_seconds",
    ):
        print(f"{name}: {metrics[name]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", type=Path, default=Path.cwd())
    prepare.add_argument("--data-root", type=Path)
    prepare.add_argument("--archive", type=Path)
    for name, layer in (("evaluate", "A"), ("holdout", "C")):
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
        command.add_argument("--seed", type=int, default=SPLIT_SEED)
        command.set_defaults(handler=evaluate, layer=layer)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        root = data_root(args.repo_root.resolve(), args.data_root)
        print(
            json.dumps(
                prepare_dataset(root, archive_source=args.archive),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
