#!/usr/bin/env python3
"""Bounded nearest-neutral pruning for one sparse residual LayerNorm site."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch


WORKSPACE = Path(__file__).resolve().parents[1]
RUN_DIR = WORKSPACE.parent
MODEL = WORKSPACE / "src" / "model.py"
TRAIN = WORKSPACE / "src" / "train.py"

SITES = {
    "block0-ln1-bias": ("blocks.0.ln1", "BLOCK0_LN1_ZERO_INDICES", "bias"),
    "block0-ln1-scale": ("blocks.0.ln1", "BLOCK0_LN1_FIXED_WEIGHT_INDICES", "scale"),
    "block1-ln1-bias": ("blocks.1.ln1", "BLOCK1_LN1_ZERO_INDICES", "bias"),
    "block1-ln1-scale": ("blocks.1.ln1", "BLOCK1_LN1_FIXED_WEIGHT_INDICES", "scale"),
    "block1-ln2-bias": ("blocks.1.ln2", "BLOCK1_LN2_ZERO_INDICES", "bias"),
    "block1-ln2-scale": ("blocks.1.ln2", "BLOCK1_LN2_FIXED_WEIGHT_INDICES", "scale"),
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def state() -> dict:
    return json.loads((RUN_DIR / "STATE.json").read_text(encoding="utf-8"))


def retained_accuracy(current: dict) -> float:
    wanted = current["incumbent"]["attempt_id"]
    if "/" in wanted:
        macro, micro = wanted.split("/", 1)
        for row in reversed(rows(RUN_DIR / "AUTOMATION_RESULTS.tsv")):
            if row["macro_attempt_id"] == macro and row["micro_attempt_id"] == micro:
                return float(row["accuracy"].rstrip("%"))
    for row in reversed(rows(RUN_DIR / "RESULTS.tsv")):
        if row["attempt_id"] == wanted:
            return float(row["accuracy"].rstrip("%"))
    raise RuntimeError(f"could not locate incumbent accuracy for {wanted}")


def latest_family_evidence() -> tuple[str, str]:
    accepted = "attempt-0059 at 2,118 parameters and 99.05%"
    failed = "attempt-0061 (discard at 54.19%)"
    for row in rows(RUN_DIR / "RESULTS.tsv"):
        if "Family: scalar pruning" not in row["proposal"]:
            continue
        if row["status"] == "keep":
            accepted = f"{row['attempt_id']} at {row['parameters']} parameters and {row['accuracy']}"
        elif row["status"] in {"discard", "error"}:
            failed = f"{row['attempt_id']} ({row['status']}, {row['accuracy'] or 'no score'})"
    return accepted, failed


def ensure_scaffold() -> None:
    source = MODEL.read_text(encoding="utf-8")
    if "BLOCK0_LN1_ZERO_INDICES" not in source:
        marker = "BLOCK0_LN2_BIAS_TIED_PAIRS = (\n    (9, 6),\n)\n"
        constants = marker + """

BLOCK0_LN1_ZERO_INDICES = (7, 8)
BLOCK0_LN1_FIXED_WEIGHT_INDICES = ()
BLOCK1_LN1_ZERO_INDICES = (4, 5, 6, 10)
BLOCK1_LN1_FIXED_WEIGHT_INDICES = (11,)
BLOCK1_LN2_ZERO_INDICES = (9, 11)
BLOCK1_LN2_FIXED_WEIGHT_INDICES = (12,)
"""
        if marker not in source:
            raise RuntimeError("could not locate normalization constant insertion point")
        source = source.replace(marker, constants, 1)
        old = """        self.ln1 = SparseBiasLayerNorm(
            d_model,
            zero_indices=((7, 8) if block_index == 0 else (4, 5, 6, 10)),
            fixed_weight_indices=(() if block_index == 0 else (11,)),
        )
"""
        new = """        self.ln1 = SparseBiasLayerNorm(
            d_model,
            zero_indices=(BLOCK0_LN1_ZERO_INDICES if block_index == 0
                          else BLOCK1_LN1_ZERO_INDICES),
            fixed_weight_indices=(BLOCK0_LN1_FIXED_WEIGHT_INDICES
                                  if block_index == 0
                                  else BLOCK1_LN1_FIXED_WEIGHT_INDICES),
        )
"""
        if old not in source:
            raise RuntimeError("could not locate ln1 constructor")
        source = source.replace(old, new, 1)
        old = """        self.ln2 = (SparseBiasLayerNorm(
                        d_model, zero_indices=(9, 11), fixed_weight_indices=(12,))
"""
        new = """        self.ln2 = (SparseBiasLayerNorm(
                        d_model, zero_indices=BLOCK1_LN2_ZERO_INDICES,
                        fixed_weight_indices=BLOCK1_LN2_FIXED_WEIGHT_INDICES)
"""
        if old not in source:
            raise RuntimeError("could not locate block-1 ln2 constructor")
        source = source.replace(old, new, 1)
        MODEL.write_text(source, encoding="utf-8")

    training = TRAIN.read_text(encoding="utf-8")
    if "sparse normalization coordinate" not in training:
        marker = """        if ('pos_emb.values' in incumbent_state
"""
        migration = """        # Exact migration after fixing one sparse normalization coordinate.
        for norm_prefix, norm_module in (
                ('blocks.0.ln1', model.blocks[0].ln1),
                ('blocks.1.ln1', model.blocks[1].ln1),
                ('blocks.1.ln2', model.blocks[1].ln2)):
            old_weight_count = (incumbent_state[norm_prefix + '.weight'].numel()
                                if norm_prefix + '.weight' in incumbent_state
                                else incumbent_state[norm_prefix + '.weight_values'].numel())
            new_weight_count = (norm_module.weight.numel()
                                if norm_module.weight is not None
                                else norm_module.weight_values.numel())
            old_bias_count = incumbent_state[norm_prefix + '.bias_values'].numel()
            new_bias_count = norm_module.bias_values.numel()
            if new_weight_count < old_weight_count or new_bias_count < old_bias_count:
                candidate_state = model.state_dict()
                for key in candidate_state:
                    if (not key.startswith(norm_prefix + '.') and key in incumbent_state
                            and candidate_state[key].shape == incumbent_state[key].shape):
                        candidate_state[key] = incumbent_state[key]
                if norm_prefix + '.weight' in incumbent_state:
                    flat_weight = incumbent_state[norm_prefix + '.weight']
                else:
                    flat_weight = incumbent_state[norm_prefix + '.weight_values'].new_ones(
                        norm_module.d_model)
                    flat_weight = flat_weight.scatter(
                        0, incumbent_state[norm_prefix + '.weight_indices'],
                        incumbent_state[norm_prefix + '.weight_values'])
                flat_bias = incumbent_state[norm_prefix + '.bias_values'].new_zeros(
                    norm_module.d_model)
                flat_bias = flat_bias.scatter(
                    0, incumbent_state[norm_prefix + '.bias_indices'],
                    incumbent_state[norm_prefix + '.bias_values'])
                if norm_module.weight is not None:
                    candidate_state[norm_prefix + '.weight'] = flat_weight
                else:
                    candidate_state[norm_prefix + '.weight_values'] = flat_weight.index_select(
                        0, norm_module.weight_indices)
                candidate_state[norm_prefix + '.bias_values'] = flat_bias.index_select(
                    0, norm_module.bias_indices)
                model.load_state_dict(candidate_state)
                torch.save({
                    'model_state': model.state_dict(),
                    'step': incumbent.get('step', 0),
                    'config': cfg,
                    'test_acc': incumbent.get('test_acc', 0.0),
                    'n_params': n_params,
                }, os.path.join(ckpt_dir, 'best.pt'))
                print("Saved checkpoint after fixing sparse normalization coordinate")
                return
"""
        if marker not in training:
            raise RuntimeError("could not locate training migration insertion point")
        TRAIN.write_text(training.replace(marker, migration + marker, 1), encoding="utf-8")


def dense_values(prefix: str, kind: str) -> torch.Tensor:
    checkpoint = torch.load(
        RUN_DIR / "state" / "incumbent.pt", map_location="cpu", weights_only=False
    )["model_state"]
    if kind == "bias":
        dense = checkpoint[prefix + ".bias_values"].new_zeros(16)
        return dense.scatter(0, checkpoint[prefix + ".bias_indices"],
                             checkpoint[prefix + ".bias_values"])
    if prefix + ".weight" in checkpoint:
        return checkpoint[prefix + ".weight"]
    dense = checkpoint[prefix + ".weight_values"].new_ones(16)
    return dense.scatter(0, checkpoint[prefix + ".weight_indices"],
                         checkpoint[prefix + ".weight_values"])


def configured_indices(constant: str) -> tuple[int, ...]:
    source = MODEL.read_text(encoding="utf-8")
    start = source.index(constant + " = (") + len(constant + " = (")
    end = source.index(")", start)
    body = source[start:end].strip()
    return tuple(int(piece.strip()) for piece in body.split(",") if piece.strip())


def add_index(constant: str, coordinate: int) -> None:
    source = MODEL.read_text(encoding="utf-8")
    start = source.index(constant + " = (") + len(constant + " = (")
    end = source.index(")", start)
    existing = source[start:end].strip()
    separator = " " if not existing or existing.endswith(",") else ", "
    replacement = existing + separator + f"{coordinate},"
    MODEL.write_text(source[:start] + replacement + source[end:], encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("site", choices=sorted(SITES))
    args = parser.parse_args()
    prefix, constant, kind = SITES[args.site]

    while True:
        current = state()
        active = current["active_automation"]
        used = int(active["micro_attempts_used"])
        cap = int(active["max_micro_trials"])
        if used >= cap:
            print(f"Reached declared micro-trial cap ({cap}).", flush=True)
            return 0
        ensure_scaffold()
        fixed = set(configured_indices(constant))
        values = dense_values(prefix, kind)
        target = 0.0 if kind == "bias" else 1.0
        options = sorted((abs(float(values[i]) - target), i, float(values[i]))
                         for i in range(16) if i not in fixed)
        if not options:
            print(f"No eligible {args.site} coordinate remains.", flush=True)
            return 0
        difference, coordinate, value = options[0]
        alternative = options[1][0] if len(options) > 1 else float("nan")
        accepted, failed = latest_family_evidence()
        parameters = int(current["incumbent"]["parameters"])
        accuracy = retained_accuracy(current)
        add_index(constant, coordinate)
        description = (f"Fix {args.site} coordinate {coordinate} to "
                       f"{'zero' if kind == 'bias' else 'one'}")
        proposal = (
            f"Family: scalar pruning. Current retained frontier is {parameters:,} parameters "
            f"at {accuracy:.6f}%, a +{accuracy - 99:.6f} percentage-point margin over 99%. "
            f"There were 23 prior macro-attempts in this family before this automation and "
            f"{used} prior micro-trials in this policy. The most recent accepted result is "
            f"{accepted}; the most recent failed result is {failed}. This is a localized "
            f"parameterization-preserving prune at {prefix}: fix {kind} coordinate {coordinate} "
            f"from {value:.10f} to its neutral value {target:.1f} (distance {difference:.3e}), "
            f"yielding {parameters - 1:,} parameters. It is more informative than the next "
            f"eligible coordinate (distance {alternative:.3e}) because it is the least checkpoint "
            f"perturbation at this isolated normalization site. The helper orders remaining "
            f"coordinates by current distance to neutral; the official runner alone accepts or "
            f"rolls back, and the first discard or error closes the path."
        )
        completed = subprocess.run(
            [sys.executable, str(RUN_DIR / "run_attempt.py"), "--run-dir", str(RUN_DIR),
             "automation-attempt", "--description", description, "--proposal", proposal],
            cwd=WORKSPACE, capture_output=True, text=True)
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            return completed.returncode
        result = rows(RUN_DIR / "AUTOMATION_RESULTS.tsv")[-1]
        if result["status"] in {"discard", "error"}:
            print(f"Recorded {result['status']} closes {args.site} path.", flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
