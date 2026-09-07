"""Review the retained B03-C3 Fashion residuals against queue-bound parents.

The campaign events for this batch name parent candidate directories that were
not retained.  The frozen review queue instead records an exact parent source
hash.  Each hash below resolves to one retained candidate directory in the
same run, so this catalog binds the comparison to source, never to a guessed
event edge.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

from experiments.generate_fashion_candidate_catalog_v2 import (
    WORKSPACE,
    complete_fingerprint,
    model_parts,
    source,
)
from experiments.ontology_review_vision_lm import FASHION, digest
from experiments.review_ontology_sources import CORE, TASK_KEYS, analyze


ROOT = WORKSPACE / "data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign/runs"
RUN = "fashion-mnist-openevolve-v2-1-mps-fashion_mnist_source_only_mps-openevolve-b03-c3"
QUEUE = WORKSPACE / "outputs/ontology/live/queue/openevolve_v21_fashion_mnist.json"
REVIEW = WORKSPACE / "outputs/ontology/openevolve_v21_fashion_mnist.json"
OUT = WORKSPACE / "experiments/ontology_fashion_candidate_references_v3_unresolved.json"

# These are the retained source directories that match the frozen queue's
# parent source hashes.  They are intentionally not the stale event parent
# identifiers for these records.
PARENTS = {
    "fc7a314": "10278e7ce7197c743db9e774c0c49753e861ee1613b10d3e03109a8643e5b99f",
    "f3ccf8": "ef114aef3fea43d70b2b879d26305d694ea713ab871f1925d19e13a4412dbfe9",
    "ecd389": "8f77cf09d4e6026f6553029eb00822d526782e680272c5ff094bda772cce1641",
    "a73389": "cfdb7321f6b6d14cb2f40799f8b05ff32e5cb8cdd29f4dd3b48b33f80053b3b2",
    "ece1f9": "88e5bd76fae41a80806b1600625c982923307ee4db4fc3f113b7b2e09d5d6175",
    "4f1d9a": "88e5bd76fae41a80806b1600625c982923307ee4db4fc3f113b7b2e09d5d6175",
    "5fb754": "7cdd182bf35de76050529ba8d85735d35b8c73d48234f6efae9d59ef01a11c68",
    "64d884": "3d242de1e129ce6babe066cfe570b677914fdfb6ca420a02e0edc79d0a0f3812",
    "487105": "795c316823f918d675f3001ab4cd122d993e21acb982fbdf0f8f8eb99ad24737",
    "698603": "7cdd182bf35de76050529ba8d85735d35b8c73d48234f6efae9d59ef01a11c68",
    "7a1a48": "8a310471037d06beab5799abb41d8f34f09900291b2bca552abdb4ff3ea00c82",
    "a78982": "7cdd182bf35de76050529ba8d85735d35b8c73d48234f6efae9d59ef01a11c68",
    "5d31ab": "71e253b404e837863a10e6288ecc48e6f9d3ff88a46195c24c76b1fa198298fb",
    "2bf526": "ea03f5b0c5bb9ce62f97a02c8185451065084009c6dd9070accf3ccacdd6fc03",
}

CHANGING = {
    "fc7a314":
        (["routing", "channel_interaction"], "adds input-dependent row and column context gates"),
    "2bf526":
        (["input_transform"], "adds a sixth, broad local-contrast image-basis channel"),
}


def base_fingerprint(child_id: str) -> dict[str, str]:
    image_basis = (
        "six fixed image-derived channels: raw image, local contrast, broad local contrast, horizontal gradient, vertical gradient and edge energy"
        if child_id.startswith("2bf526") else
        "five fixed image-derived channels: raw image, local contrast, horizontal gradient, vertical gradient and edge energy"
    )
    fp = {
        "input_units": "grayscale image pixels",
        "input_transform": image_basis,
        "embedding": "not applicable: continuous-valued pixels enter fixed image-basis arithmetic and convolutions directly",
        "position": "implicit local image neighborhoods; no explicit positional embedding",
        "mixing": "dense Conv2d backbone with local depthwise-plus-pointwise convolutional refinement",
        "routing": "fixed source-defined tensor routing and a fixed reflected-translation/flip inference ensemble",
        "state": "no persistent recurrent feature state",
        "feedforward": "no separate transformer feedforward block",
        "parameter_construction": "free learned convolutional and affine parameters",
        "sharing": "no explicit parameter tying beyond source-defined module reuse",
        "normalization": "BatchNorm2d feature normalization and LayerNorm in the affine classifier",
        "connectivity": "fixed convolutional feedforward stages with additive local refinement",
        "aggregation": "flattened final spatial feature grid",
        "output": "learned affine ten-class logits",
        "symmetry": "no explicit equivariant or canonicalized construction",
        "conditional_compute": "fixed architecture; no input-dependent early exit or experts",
        "activation": "GELU",
        "stochasticity": "Dropout in the classifier during training",
        "bottleneck": "no separate factorized bottleneck",
        "iteration": "fixed feedforward stages",
        "other": "complete candidate and queue-bound retained-parent train.py sources were directly read and diffed",
        "spatial_units": "image grid",
        "spatial_operator": "dense and depthwise local Conv2d operators",
        "scale_representation": "two MaxPool2d reductions followed by a fixed-resolution convolutional feature grid",
        "spatial_readout": "flattened spatial grid through an affine classifier",
        "channel_interaction": "dense convolution and 1x1 pointwise convolution",
        "spatial_downsampling": "two MaxPool2d(2) operations",
    }
    if child_id.startswith("fc7a314"):
        fp.update(
            mixing="dense/depthwise Conv2d plus 1x1 projections for coordinate gates",
            routing="input-dependent global row and column means drive separate tanh channel scales",
            connectivity="local convolutional backbone and refinement followed by multiplicative row/column coordinate gates",
            activation="GELU and tanh coordinate gates",
            channel_interaction="1x1 coordinate reduction and separate 1x1 row/column gate projections",
        )
    elif child_id.startswith("f3ccf"):
        fp["connectivity"] = "fixed convolutional stages with two additive local depthwise-plus-pointwise refinement blocks"
    elif child_id.startswith("a733"):
        fp.update(
            connectivity="fixed convolutional stages followed by a learned stride-2 local aggregate before the affine classifier",
            scale_representation="two MaxPool2d reductions followed by a learned 3x3 stride-2 aggregate",
            spatial_downsampling="two MaxPool2d(2) operations and one learned stride-2 Conv2d",
        )
    elif child_id.startswith("ece1"):
        fp.update(
            connectivity="parallel learned stride-2 local aggregate and fixed average-pool paths concatenate before the affine classifier",
            aggregation="concatenated learned local and fixed average-pooled spatial grids",
            scale_representation="two MaxPool2d reductions followed by parallel learned stride-2 and average-pool grids",
            spatial_downsampling="two MaxPool2d(2), one learned stride-2 Conv2d and one fixed average pool",
        )
    elif child_id.startswith(("4f1d", "64d884", "2bf526")):
        fp.update(
            connectivity="parallel local and dilated stride-2 convolution branches concatenate before affine classification",
            aggregation="concatenated local and dilated-context spatial grids",
            scale_representation="two MaxPool2d reductions followed by parallel local and dilated stride-2 grids",
            spatial_downsampling="two MaxPool2d(2) operations and parallel stride-2 local/dilated Conv2d branches",
            channel_interaction="dense, depthwise and 1x1 local convolutional projections",
        )
    elif child_id.startswith("5fb754"):
        fp["spatial_operator"] = "dense and depthwise local Conv2d operators; the depthwise dilated operator uses replicate boundary padding"
    elif child_id.startswith("5d31"):
        fp["state"] = "EMA parameter shadow is updated after optimizer steps and copied for evaluation; no persistent recurrent feature state"
    return fp


def family_signature(child_id: str) -> dict[str, str]:
    signature = {
        "task": "fashion",
        "input": "fixed five-channel local image basis",
        "mixing": "local dense/depthwise convolution",
        "routing": "fixed local tensor flow and fixed transformed-view ensemble",
        "readout": "flattened spatial affine classifier",
    }
    if child_id.startswith("fc7a314"):
        signature["routing"] = "input-dependent row/column coordinate channel gates"
    if child_id.startswith("2bf526"):
        signature["input"] = "fixed six-channel image basis including broad local contrast"
    return signature


def inference(child_id: str) -> dict[str, str]:
    value = "original and horizontal-flip logits at reflected translated image positions use fixed source-defined weights"
    if child_id.startswith("ecd389"):
        value = "fixed transformed-view probabilities are weighted, clamped and logged"
    if child_id.startswith(("698603", "7a1a", "a789")):
        value = "fixed transformed-view logit weights vary by image-grid offset but not by input content"
    if child_id.startswith("5d31"):
        value += "; evaluation first installs the fixed EMA parameter shadow"
    return {"procedure": value}


def main() -> None:
    queue = {(row["candidate_id"], row["source_sha256"]): row
             for row in json.loads(QUEUE.read_text(encoding="utf-8"))}
    previous = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"unresolved": []}
    # The source identities, rather than a mutable current classification,
    # select this finite review batch.  The generator must remain reproducible
    # after the staged declarations have cleared the queue.
    residuals = [row for row in json.loads(REVIEW.read_text(encoding="utf-8"))["rows"]
                 if row["run_id"] == RUN
                 and any(row["candidate_id"].startswith(prefix) for prefix in PARENTS)]
    children = {prefix: next((row["candidate_id"] for row in residuals
                              if row["candidate_id"].startswith(prefix)), None)
                for prefix in PARENTS}
    if len(residuals) != len(PARENTS) or any(child is None for child in children.values()):
        raise ValueError("the audited B03-C3 residual group changed")
    profiles = []
    required = CORE + TASK_KEYS["fashion"].split()
    for prefix, parent_id in PARENTS.items():
        child_id = children[prefix]
        child = source(ROOT / RUN / "candidates" / child_id)
        parent = source(ROOT / RUN / "candidates" / parent_id)
        child_hash, parent_hash = digest(child), digest(parent)
        queue_row = queue.get((child_id, child_hash))
        if not queue_row or queue_row["parent_source_sha256"] != [parent_hash]:
            raise ValueError("queue-bound parent source does not match retained source for " + child_id)
        analysis = analyze(child, "fashion")
        if analysis["error"]:
            raise ValueError(analysis["error"])
        fingerprint = base_fingerprint(child_id)
        missing = [key for key in required if not fingerprint.get(key)]
        if missing:
            raise ValueError("incomplete fingerprint for " + child_id + ": " + repr(missing))
        classification, changed, reason = "preserving", [], "ordinary local-CNN, training-lifecycle or fixed-inference setting change"
        if prefix in CHANGING:
            changed, reason = CHANGING[prefix]
            classification = "changing"
        diff = list(difflib.unified_diff(parent["train.py"].splitlines(), child["train.py"].splitlines(), lineterm=""))
        profiles.append({
            "source_sha256": child_hash,
            "program_sha256": digest(model_parts(child)),
            "program_parts": model_parts(child),
            "reference_run": RUN,
            "reference_proposal": queue_row["proposal"],
            "candidate_id": child_id,
            "parent_candidate_id": parent_id,
            "parent_source_sha256": parent_hash,
            "transition_classification": classification,
            "changed_components": changed,
            "fingerprint": fingerprint,
            "family_signature": family_signature(child_id),
            "family_label": "coordinate-gated local CNN" if child_id.startswith("fc7a314") else "six-channel image-basis local CNN" if child_id.startswith("2bf526") else "fixed-image-basis local CNN",
            "training": analysis["training"],
            "inference": inference(child_id),
            "settings": analysis["settings"],
            "notes": "Complete child and queue-hash-resolved retained parent sources were directly read and diffed. " + reason + ".",
            "evidence": [{
                "kind": "complete exact-program queue-bound parent-child diff review",
                "source_sha256": child_hash,
                "parent_source_sha256": parent_hash,
                "candidate_id": child_id,
                "retained_parent_candidate_id": parent_id,
                "queue_parent_source_sha256": queue_row["parent_source_sha256"],
                "changed_lines": {"added": sum(line.startswith("+") and not line.startswith("+++") for line in diff),
                                  "removed": sum(line.startswith("-") and not line.startswith("---") for line in diff)},
                "classification": classification,
                "changed_components": changed,
                "reason": reason,
                "source_path": str(ROOT / RUN / "candidates" / child_id / "train.py"),
                "parent_source_path": str(ROOT / RUN / "candidates" / parent_id / "train.py"),
            }],
        })
    previous.update({"version": "fashion-candidate-source-review-v3-b03c3-readable-residuals", "reviewed_profiles": profiles})
    OUT.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
