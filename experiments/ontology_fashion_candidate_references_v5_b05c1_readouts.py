"""Closed, exact-parent B05-C1 supplementary-readout profiles.

This finite catalog contains only the three SHA-bound child/parent pairs below.
It never derives a classification for a nearby source or a source-only match.
"""
from __future__ import annotations

import difflib
import json

from experiments.generate_fashion_candidate_catalog_v2 import ROOT, source
from experiments.ontology_review_vision_lm import digest
from experiments.review_ontology_sources import CORE, TASK_KEYS


RUN = "fashion-mnist-openevolve-v2-1-mps-fashion_mnist_source_only_mps-openevolve-b05-c1"
SPECS = {
    "e3b2c1de2209c0c0a3d360e72e95a43e96d6c640b9b76aa8f04537fa0d61f967": (
        "45c6d3fa8e4e3228e1a7737ea549fab13ca49b2df2d0ad37bf4c5ed533138a9d",
        "449ff3b29d782ca7a6797aeb1abe116f33167f2e9d320d61bc4ea105483902b7",
        "c5624895305aa31458c94593a6af8ac5a6f8ba8d038d86b0cf7df1326a65b3ea",
        ["aggregation", "scale_representation", "spatial_readout"],
        "adds an affine 2x2 adaptive-average-pooled convolutional coarse-logit head",
    ),
    "215b7237e3df6a950eea6604a1015f4a39a0a1ececc4a7fcee7b40f9e591d7a1": (
        "8e29cdf23f69da90deecfca7f4dfc336f28cb5027cf2b6832554fbbf18de5909",
        "7f3f096cc746621d1524e3ffe34d0fd6c36e5f57328507eba6e546a861128638",
        "e03fcaf98ecf2d0fb0315396cdc4c166a0465065e02a997e26f01adccdd104eb",
        ["input_transform", "aggregation", "scale_representation", "spatial_readout"],
        "adds an affine 2x2-average-pooled raw-image silhouette logit branch",
    ),
    "7033915b2ba004743d1f9a4459e4a33d1c664dabe901bfc50d464fec0ddf55b9": (
        "bce4135b2ee456027640bcd8a2e9acf72e764535b2b2192ee8afdbb88a0989aa",
        "bcbd1992bd8eeb3576a2abd5a2ece586413f22264dd1b33cfb7bf0bb83719839",
        "b434225f58b721a6ea616e54e4c61c92f7c0dc6760b513ef6d77fc7df301d819",
        ["aggregation", "scale_representation", "spatial_readout"],
        "replaces the flattened feature head with fixed 1x1, 2x2 and 4x4 adaptive-average spatial-pyramid readout",
    ),
}

PROPOSALS = {sha: proposal for sha, proposal in zip(SPECS, (51, 64, 100))}

BASE = {
    "input_units": "grayscale image pixels",
    "input_transform": "identity image tensor input",
    "embedding": "not applicable: continuous-valued grayscale pixels enter convolutional operators directly",
    "position": "implicit local image neighborhoods; no explicit positional embedding",
    "mixing": "dense local Conv2d residual blocks",
    "routing": "fixed convolutional residual flow and fixed translated/flip inference views",
    "state": "no persistent recurrent feature state",
    "feedforward": "no separate transformer feedforward block",
    "parameter_construction": "free learned convolutional and affine parameters",
    "sharing": "residual-block modules are reused only as explicitly called",
    "normalization": "BatchNorm2d feature normalization and BatchNorm1d readout normalization",
    "connectivity": "fixed convolutional residual stages with parallel affine logit heads summed at output",
    "aggregation": "flattened convolutional grid plus mean, standard-deviation and maximum feature statistics",
    "output": "summed learned affine ten-class logits",
    "symmetry": "fixed horizontal-flip and translated-view inference ensemble; no equivariant parameter construction",
    "conditional_compute": "fixed architecture; no input-dependent early exit or experts",
    "activation": "GELU",
    "stochasticity": "Dropout in the flattened affine classifier during training",
    "bottleneck": "no separate factorized bottleneck",
    "iteration": "fixed feedforward stages",
    "other": "complete exact retained parent and child train.py sources were directly read and diffed",
    "spatial_units": "image grid",
    "spatial_operator": "dense local Conv2d",
    "scale_representation": "two MaxPool2d reductions followed by a 7x7 convolutional grid",
    "spatial_readout": "flattened spatial affine logits plus affine mean/std/max statistic logits",
    "channel_interaction": "dense convolution and affine statistic heads",
    "spatial_downsampling": "two MaxPool2d(2) operations",
}

OVERRIDES = {
    "e3b2c1de2209": {
        "aggregation": "flattened convolutional grid, mean/std/max statistics, and a flattened 2x2 adaptive-average-pooled grid",
        "scale_representation": "two MaxPool2d reductions followed by 7x7 and AdaptiveAvgPool2d(2,2) grids",
        "spatial_readout": "flattened-grid and statistic logits summed with a 2x2 adaptive-average-pooled affine coarse head",
    },
    "215b7237e3df": {
        "input_transform": "identity image tensor plus a fixed 2x2 average-pooled raw-image silhouette readout branch",
        "routing": "fixed convolutional residual flow, fixed raw-image silhouette branch, and fixed translated/flip inference views",
        "aggregation": "flattened convolutional grid, mean/std/max statistics, and a flattened 14x14 average-pooled raw-image silhouette",
        "scale_representation": "two MaxPool2d reductions plus a fixed 2x2 raw-image average-pool silhouette",
        "spatial_readout": "flattened-grid and statistic logits summed with an average-pooled raw-image silhouette affine head",
    },
    "7033915b2ba0": {
        "mixing": "dense local Conv2d residual blocks plus global-mean sigmoid channel recalibration",
        "routing": "global feature mean drives a sigmoid channel scale; translated/flip inference views remain fixed",
        "connectivity": "fixed convolutional residual stages followed by channel recalibration and parallel statistic/spatial-pyramid affine logits",
        "aggregation": "mean/std/max statistics plus concatenated 1x1, 2x2 and 4x4 adaptive-average-pooled feature grids",
        "activation": "GELU and sigmoid channel gate",
        "scale_representation": "two MaxPool2d reductions plus fixed AdaptiveAvgPool2d 1x1, 2x2 and 4x4 pyramid grids",
        "spatial_readout": "concatenated adaptive-average spatial-pyramid affine logits plus statistic logits",
        "channel_interaction": "global-mean learned per-channel sigmoid recalibration",
    },
}


def _fingerprint(source_sha256: str) -> dict[str, str]:
    if source_sha256 not in SPECS:
        raise KeyError("unreviewed Fashion source: " + source_sha256)
    result = dict(BASE)
    result.update(OVERRIDES[source_sha256[:12]])
    return result


def profiles() -> list[dict]:
    candidates = ROOT / RUN / "candidates"
    rows = []
    for source_sha256, (candidate_id, parent_sha256, parent_id, changed, reason) in SPECS.items():
        child, parent = source(candidates / candidate_id), source(candidates / parent_id)
        if digest(child) != source_sha256 or digest(parent) != parent_sha256:
            raise ValueError("source binding changed for " + candidate_id)
        fingerprint = _fingerprint(source_sha256)
        missing = [key for key in CORE + TASK_KEYS["fashion"].split() if not fingerprint.get(key)]
        if missing:
            raise ValueError("incomplete fingerprint: " + repr(missing))
        diff = list(difflib.unified_diff(parent["train.py"].splitlines(), child["train.py"].splitlines(), lineterm=""))
        family = {"task": "fashion", "mixing": "local residual convolution", "routing": "fixed local flow and transformed-view ensemble", "readout": "statistic-augmented spatial affine logits"}
        if source_sha256.startswith("215b"):
            family["readout"] = "statistic-augmented spatial and raw-image silhouette affine logits"
        if source_sha256.startswith("7033"):
            family["routing"] = "global-mean sigmoid channel gate with fixed transformed-view ensemble"
            family["readout"] = "statistic-augmented adaptive-average spatial-pyramid affine logits"
        rows.append({
            "source_sha256": source_sha256, "reference_run": RUN, "reference_proposal": PROPOSALS[source_sha256],
            "candidate_id": candidate_id, "parent_candidate_id": parent_id, "parent_source_sha256": parent_sha256,
            "transition_classification": "changing", "changed_components": changed, "fingerprint": fingerprint,
            "family_signature": family, "family_label": "residual CNN with supplementary readouts",
            "training": {"optimizer": "AdamW (declared)", "objective": "cross entropy with label smoothing (declared)", "augmentation": "translated and horizontally flipped image views (declared)"},
            "inference": {"procedure": "fixed original/translated horizontal-flip probability ensemble with positive scalar log calibration"},
            "notes": "Complete exact retained parent and child train.py sources were directly read and diffed. " + reason + ".",
            "evidence": [{"kind": "complete exact-program queue-bound parent-child diff review", "source_sha256": source_sha256,
                          "parent_source_sha256": parent_sha256, "candidate_id": candidate_id,
                          "retained_parent_candidate_id": parent_id, "directed_parent_source_sha256": parent_sha256,
                          "classification": "changing", "changed_components": changed, "reason": reason,
                          "changed_lines": {"added": sum(line.startswith("+") and not line.startswith("+++") for line in diff),
                                            "removed": sum(line.startswith("-") and not line.startswith("---") for line in diff)}}],
        })
    return rows

