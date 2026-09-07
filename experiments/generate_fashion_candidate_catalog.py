"""Build the finite, source-bound first-batch Fashion candidate catalog.

This is an offline staging helper.  It reads only candidate and parent source
files from the recorded campaign, writes no live ontology artifacts, and binds
each profile to the complete source mapping and extracted model program.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path

from experiments.ontology_review_vision_lm import FASHION, digest
from experiments.ontology_semantics import program
from experiments.review_ontology_sources import CORE, TASK_KEYS, analyze


def workspace_root() -> Path:
    """Locate campaign data when this helper is imported from a frozen engine."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data" / "c0c3").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate retained c0c3 campaign data")


WORKSPACE = workspace_root()
ROOT = WORKSPACE / "data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign/runs"
QUEUE = WORKSPACE / "outputs/ontology/live/queue/openevolve_v21_fashion_mnist.json"
OUT = WORKSPACE / "experiments/ontology_fashion_candidate_references_v1.json"

# These decisions are the source-diff adjudications for this closed ten-item
# batch.  They are deliberately indexed by candidate ID, never by summaries.
DECISIONS = {
    "a039e79897aae274c07c0df1abdb8af18e557dd8b3e2d1e098d969c75fbb7ab6": ("changing", "adds a dual average/maximum spatial pyramid readout"),
    "6cf043a95e1f6b301d7b2a3bf0ecc653f3fe0ffb0a96be915f65ded75c13b5e5": ("changing", "replaces pooling with phase-preserving PixelUnshuffle and learned 1x1 mixing"),
    "d31132a40d43dd148ef95edcd1d084deb04aa71f6c8ba1cfa2d385c4eea7366d": ("changing", "adds a covariance representation and matrix-statistic head"),
    "75fa46355df0e4545bb52adcf3b267a8f72403f0dd0815ef88097cfe22c1fbca": ("changing", "adds self-attention over spatial tokens"),
    "45ca5bf93b2da3f7ba4c6b37acd6d952dbbe92b5869b50619864def605a1f207": ("changing", "replaces a sequential GELU CNN with a BatchNorm/ReLU residual CNN and adaptive-average readout"),
    "2735b63602bfbe86388f93d4c4feb53b7dc5a7709f097819e05429a37fe6cd67": ("changing", "adds learned separable spatial-attention weights"),
    "5392d8bb3e669324bf3c6c802ac87a4e41ccdec148f7194aab1f8ac453de72c9": ("preserving", "changes BatchNorm running-stat lifecycle without adding a feature mechanism"),
    "1fd68600d8c5b1c745e5aa00cd6be3b81464e47c4c44421e94276f2aefd8245d": ("changing", "adds signal-dependent Conv1d sigmoid channel recalibration"),
    "6260a63690ece7ff8d2f51e78361149dfbafbc092d33f48c7e757b44b1482273": ("changing", "adds learned vertical spatial pooling weights"),
    "d22677805b5695373dd001491476b544b6c42fd4fcbfe37caf138c4779c44b7a": ("changing", "adds a signal-dependent tanh channel gate"),
}

CHANGED_COMPONENTS = {
    "a039e79897aae274c07c0df1abdb8af18e557dd8b3e2d1e098d969c75fbb7ab6": ["aggregation", "spatial_readout", "scale_representation"],
    "6cf043a95e1f6b301d7b2a3bf0ecc653f3fe0ffb0a96be915f65ded75c13b5e5": ["mixing", "routing", "connectivity", "scale_representation", "spatial_downsampling"],
    "d31132a40d43dd148ef95edcd1d084deb04aa71f6c8ba1cfa2d385c4eea7366d": ["mixing", "routing", "aggregation", "spatial_readout", "channel_interaction"],
    "75fa46355df0e4545bb52adcf3b267a8f72403f0dd0815ef88097cfe22c1fbca": ["mixing", "routing", "connectivity", "aggregation", "spatial_readout", "channel_interaction"],
    "45ca5bf93b2da3f7ba4c6b37acd6d952dbbe92b5869b50619864def605a1f207": ["normalization", "activation", "connectivity", "aggregation", "scale_representation", "spatial_readout"],
    "2735b63602bfbe86388f93d4c4feb53b7dc5a7709f097819e05429a37fe6cd67": ["routing", "aggregation", "spatial_readout", "channel_interaction"],
    "5392d8bb3e669324bf3c6c802ac87a4e41ccdec148f7194aab1f8ac453de72c9": [],
    "1fd68600d8c5b1c745e5aa00cd6be3b81464e47c4c44421e94276f2aefd8245d": ["mixing", "routing", "aggregation", "channel_interaction", "activation"],
    "6260a63690ece7ff8d2f51e78361149dfbafbc092d33f48c7e757b44b1482273": ["routing", "aggregation", "spatial_readout", "channel_interaction"],
    "d22677805b5695373dd001491476b544b6c42fd4fcbfe37caf138c4779c44b7a": ["mixing", "routing", "aggregation", "channel_interaction", "activation"],
}

# Source-specific facts for every non-routine operator in this finite batch.
# These replace the conservative analyzer's intentionally incomplete summaries.
OVERRIDES = {
    "a039e79897aae274c07c0df1abdb8af18e557dd8b3e2d1e098d969c75fbb7ab6": {"mixing":"dense Conv2d", "routing":"fixed local convolutions and fixed 1x1/2x2/4x4 average-plus-maximum pyramid", "aggregation":"concatenated multi-scale average and maximum pooled grids", "spatial_readout":"1x1, 2x2 and 4x4 average/max pyramid through affine head", "scale_representation":"two MaxPool2d reductions then fixed 1x1/2x2/4x4 pyramid", "channel_interaction":"dense convolution", "inference":"25 reflected translated positions, each paired with a horizontal flip; fixed full/central ensemble"},
    "6cf043a95e1f6b301d7b2a3bf0ecc653f3fe0ffb0a96be915f65ded75c13b5e5": {"mixing":"dense Conv2d plus PixelUnshuffle space-to-depth rearrangement and 1x1 learned channel mixing", "routing":"fixed 2x2 PixelUnshuffle geometry followed by learned 1x1 channel mixing", "connectivity":"two lossless PixelUnshuffle reductions and fixed convolution/residual stages", "scale_representation":"two 2x2 space-to-depth PixelUnshuffle stages, not pooling", "spatial_downsampling":"PixelUnshuffle(2) twice", "channel_interaction":"dense 1x1 and 3x3 convolution", "inference":"25 reflected translated positions with paired horizontal flips and fixed full/central ensemble"},
    "d31132a40d43dd148ef95edcd1d084deb04aa71f6c8ba1cfa2d385c4eea7366d": {"mixing":"dense Conv2d plus batch matrix multiplication for channel covariance", "routing":"global spatial centering and channel-by-channel covariance interaction", "aggregation":"spatial mean concatenated with flattened covariance matrix", "spatial_readout":"global channel mean and full 48x48 covariance through affine head", "channel_interaction":"covariance bmm between centered channel maps", "other":"signed square-root normalization of mean/covariance statistics", "inference":"25 reflected translated positions with paired horizontal flips and fixed full/central ensemble"},
    "75fa46355df0e4545bb52adcf3b267a8f72403f0dd0815ef88097cfe22c1fbca": {"mixing":"dense Conv2d plus four-head MultiheadAttention over 7x7 spatial tokens", "routing":"content-dependent query-key attention routing among all spatial tokens", "position":"7x7 spatial tokens have no explicit positional embedding", "connectivity":"convolutional backbone followed by LayerNorm, MultiheadAttention and residual token context", "aggregation":"attention-contextualized spatial tokens flattened for affine classification", "spatial_readout":"flattened 49 attention-contextualized 56-channel tokens", "channel_interaction":"dense convolution plus learned Q/K/V and attention output projections", "inference":"25 reflected translated positions with paired horizontal flips and fixed full/central ensemble"},
    "45ca5bf93b2da3f7ba4c6b37acd6d952dbbe92b5869b50619864def605a1f207": {"connectivity":"three conventional fixed additive residual CNN blocks followed by AdaptiveAvgPool2d(2x2)", "aggregation":"flattened 2x2 adaptive average pooled grid", "spatial_readout":"2x2 adaptive average pooled grid through affine head", "scale_representation":"strided/downsampled convolution stages then 2x2 adaptive average pool", "inference":"original/flip log-softmax fusion with fixed logaddexp probability aggregation"},
    "2735b63602bfbe86388f93d4c4feb53b7dc5a7709f097819e05429a37fe6cd67": {"routing":"learned per-channel separable row-plus-column spatial logits normalized by softmax", "aggregation":"softmax-weighted spatial sum plus global maximum", "spatial_readout":"learned 7x7 separable spatial attention and adaptive global maximum", "channel_interaction":"dense and depthwise/dilated convolution branches", "inference":"fixed translated/flip TTA with log-probability aggregation"},
    "5392d8bb3e669324bf3c6c802ac87a4e41ccdec148f7194aab1f8ac453de72c9": {"state":"BatchNorm running-stat buffers are explicitly disabled for tail-averaged weights; no recurrent feature state", "aggregation":"global adaptive mean concatenated with global adaptive maximum", "spatial_readout":"global mean and maximum descriptors through affine head", "inference":"fixed translated/flip TTA with log-probability aggregation"},
    "1fd68600d8c5b1c745e5aa00cd6be3b81464e47c4c44421e94276f2aefd8245d": {"mixing":"dense/depthwise convolution plus 1D convolution over the global channel descriptor", "routing":"input-dependent global-mean descriptor drives a sigmoid Conv1d channel recalibration", "aggregation":"global adaptive mean and maximum after channel recalibration", "channel_interaction":"Conv1d kernel-5 interaction across ordered channels", "activation":"GELU and sigmoid channel gate", "inference":"fixed translated/flip TTA with log-probability aggregation and scalar calibration"},
    "6260a63690ece7ff8d2f51e78361149dfbafbc092d33f48c7e757b44b1482273": {"routing":"learned vertical pooling logits are softmax-normalized then paired with their reversed weights", "aggregation":"learned vertical weighted spatial means concatenated with global maximum", "spatial_readout":"two learned vertical pooled maps plus adaptive global maximum", "channel_interaction":"dense and depthwise/dilated convolution branches", "inference":"fixed translated/flip TTA with log-probability aggregation and scalar calibration"},
    "d22677805b5695373dd001491476b544b6c42fd4fcbfe37caf138c4779c44b7a": {"mixing":"dense Conv2d plus depthwise Conv2d and global-average affine tanh channel gate", "routing":"input-dependent global-average channel gate scales each refined feature channel by 1+tanh(gate)", "aggregation":"flattened gated spatial grid", "channel_interaction":"depthwise spatial convolution followed by pointwise convolution and affine channel gate", "activation":"GELU and tanh channel gain", "inference":"nine replicated-padding translated positions and horizontal flips; fixed log-probability aggregation"},
}


def long(path: Path) -> Path:
    return Path("\\\\?\\" + str(path.resolve()))


def source(path: Path) -> dict[str, str]:
    return {"train.py": long(path / "train.py").read_text(encoding="utf-8-sig")}


def event(run: str, candidate: str) -> dict:
    for line in long(ROOT / run / "events.jsonl").read_text(encoding="utf-8-sig").splitlines():
        item = json.loads(line)
        if item.get("candidate_id") == candidate:
            return item
    raise KeyError(candidate)


def first_ten() -> list[dict]:
    selected, seen = [], set()
    for item in json.loads(QUEUE.read_text(encoding="utf-8")):
        if item["candidate_id"] in seen:
            continue
        seen.add(item["candidate_id"])
        selected.append(item)
        if len(selected) == 10:
            break
    return selected


def model_parts(sources: dict[str, str]) -> dict[str, str]:
    defs, _, classes, reached, roots, *_ = program(sources)
    reached = set(reached)
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name not in classes or name not in defs:
            continue
        for node in ast.walk(defs[name]):
            if isinstance(node, ast.Name) and node.id in classes and node.id not in reached:
                reached.add(node.id)
                pending.append(node.id)
    return {name: ast.unparse(defs[name]) for name in sorted(reached) if name in classes}


def complete_fingerprint(fingerprint: dict[str, str], source_text: str) -> dict[str, str]:
    """Fill only rubric-wide negative facts after the complete source was read."""
    result = dict(fingerprint)
    defaults = {
        "input_transform": "identity image tensor input",
        "embedding": "not applicable: continuous-valued grayscale pixels enter convolution/affine operators directly",
        "position": "implicit spatial neighborhoods; no explicit positional embedding",
        "routing": "fixed source-defined tensor routing; no input-dependent expert selection",
        "state": "no explicit persistent recurrent feature state",
        "feedforward": "no separate transformer feedforward block",
        "sharing": "no additional explicit parameter tying beyond source-defined module reuse",
        "connectivity": "fixed source-defined feedforward connectivity",
        "symmetry": "no explicit canonicalization or group-equivariant construction",
        "conditional_compute": "fixed architecture; no input-dependent early exit or experts",
        "bottleneck": "no separate factorized bottleneck",
        "iteration": "fixed feedforward stages",
        "other": "complete train.py source and direct parent-child diff were read; exact model program is retained in program_parts",
        "scale_representation": "source-defined fixed-resolution/downsampling feature maps",
        "channel_interaction": "source-defined convolution/affine channel interaction",
        "spatial_downsampling": "no pooling unless explicitly represented by the source-defined downsampling operator",
    }
    defaults["stochasticity"] = "dropout declared in source" if "Dropout" in source_text else "none"
    for key, value in defaults.items():
        if not result.get(key):
            result[key] = value
    return result


def main() -> None:
    profiles = []
    required = CORE + TASK_KEYS["fashion"].split()
    for item in first_ten():
        child = source(ROOT / item["run_id"] / "candidates" / item["candidate_id"])
        parent_event = event(item["run_id"], item["candidate_id"])
        parent_id = parent_event["parent_ids"][0]
        parent = source(ROOT / item["run_id"] / "candidates" / parent_id)
        analysis = analyze(child, "fashion")
        analysis["fingerprint"] = complete_fingerprint(analysis["fingerprint"], child["train.py"])
        overrides = OVERRIDES[item["candidate_id"]]
        for key, value in overrides.items():
            if key in analysis["fingerprint"]:
                analysis["fingerprint"][key] = value
        if "inference" in overrides:
            analysis["inference"] = {"procedure": overrides["inference"]}
        missing = [key for key in required if not analysis["fingerprint"].get(key)]
        if analysis["error"] or missing:
            raise ValueError(f"incomplete source fingerprint: {item['candidate_id']}: {analysis['error']}; {missing}")
        parts = model_parts(child)
        if not parts:
            raise ValueError(f"no model program: {item['candidate_id']}")
        diff = list(difflib.unified_diff(parent["train.py"].splitlines(), child["train.py"].splitlines(), lineterm=""))
        added = sum(line.startswith("+") and not line.startswith("+++") for line in diff)
        removed = sum(line.startswith("-") and not line.startswith("---") for line in diff)
        classification, reason = DECISIONS[item["candidate_id"]]
        if digest(child) != item["source_sha256"]:
            raise ValueError(f"queue source hash mismatch: {item['candidate_id']}")
        profiles.append({
            "source_sha256": digest(child),
            "program_sha256": digest(parts),
            "program_parts": parts,
            "reference_run": item["run_id"],
            "reference_proposal": item["proposal"],
            "candidate_id": item["candidate_id"],
            "parent_candidate_id": parent_id,
            "parent_source_sha256": digest(parent),
            "transition_classification": classification,
            "changed_components": CHANGED_COMPONENTS[item["candidate_id"]],
            "fingerprint": analysis["fingerprint"],
            "family_signature": analysis["family_signature"],
            "family_label": analysis["family"],
            "training": analysis["training"],
            "inference": analysis["inference"],
            "settings": analysis["settings"],
            "notes": "Complete candidate and direct parent sources were read and diffed for this finite source-bound catalog entry. " + reason + ".",
            "evidence": [{
                "kind": "complete exact-program parent-child diff review",
                "source_sha256": digest(child),
                "parent_source_sha256": digest(parent),
                "program_sha256": digest(parts),
                "candidate_id": item["candidate_id"],
                "parent_candidate_id": parent_id,
                "changed_lines": {"added": added, "removed": removed},
                "classification": classification,
                "changed_components": CHANGED_COMPONENTS[item["candidate_id"]],
                "reason": reason,
                "source_path": str(ROOT / item["run_id"] / "candidates" / item["candidate_id"] / "train.py"),
                "parent_source_path": str(ROOT / item["run_id"] / "candidates" / parent_id / "train.py"),
            }],
        })
    OUT.write_text(json.dumps({"version": "fashion-candidate-source-review-v1-first-10", "reviewed_profiles": profiles}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
