"""Closed, SHA-bound B04-C1 Fashion source profiles.

The ten declarations below were assigned by directly reading each complete
child program and its queue-hash-resolved parent.  This module never matches
an unlisted source or derives a decision from a general program shape.
"""
from __future__ import annotations

import difflib
from pathlib import Path

from experiments.generate_fashion_candidate_catalog_v2 import ROOT, source
from experiments.ontology_review_vision_lm import digest
from experiments.review_ontology_sources import CORE, TASK_KEYS

RUN = "fashion-mnist-openevolve-v2-1-mps-fashion_mnist_source_only_mps-openevolve-b04-c1"

# child SHA: candidate id, parent SHA, retained parent candidate id,
# transition result, changed components, direct source-diff conclusion.
SPECS = {
"d2016695edf7fbcae46a243989a90210a430ddf7bf6035e77b81606b342f77e6": ("98aa3989c930175414538943daca0ea107bb536f4de519f7b74e547ffda96ff8", "ba3c2cb6958beace093d5e60af527c3a22412a0c61742ee1f4d7b0cd0f33f410", "971307c9e125610d788d68b5c772c5b8b4870f54143b04ed40684d9c5e528437", "preserving", [], "adds fixed local depthwise near/wide context branches before the affine head"),
"097a2f73ec4f766442638a9907b22d9d6a943b923e9c0bc0e065463481ab959d": ("f353893c843d6d87c6d50468849190321a86bd49126d9d72253c5029364d3b14", "ba3c2cb6958beace093d5e60af527c3a22412a0c61742ee1f4d7b0cd0f33f410", "971307c9e125610d788d68b5c772c5b8b4870f54143b04ed40684d9c5e528437", "changing", ["aggregation", "spatial_readout", "scale_representation"], "adds a learned adaptive-average-pooled spatial skip logit readout"),
"19945e1c3924741574d95f789d70455c04e5edfbc20d5c871a26ea21f154d369": ("44a0f2eceb77ed778ddd8e8aba036ae1a936b9dd3ea77fd2a12341b26d2cdd56", "9563e71cb448a17e4514306de879d86ca4de3b15bfbab9ce86170719b17140f1", "a1ad848d0ab26566aefefe07fc7ff9c514ae70c8ad6cca9c4aea2ad3e9d39056", "changing", ["routing", "aggregation", "channel_interaction"], "adds a global paired-view average descriptor driving a tanh channel gate"),
"d05ae9c873b37373a916d18ba7fd36a7df33832881cdcf7d6003b8afc6728bd6": ("f604469fe1c1dbd196ecde4f7b9ad6e0c7319557a659272282e674cf47a784ec", "9563e71cb448a17e4514306de879d86ca4de3b15bfbab9ce86170719b17140f1", "a1ad848d0ab26566aefefe07fc7ff9c514ae70c8ad6cca9c4aea2ad3e9d39056", "changing", ["mixing", "routing", "aggregation", "spatial_readout", "channel_interaction"], "adds a signed square-root normalized bilinear spatial statistic head"),
"fba2269f217c464359f22feccd853e26e5fd92b61794b789b94ffaef24c69027": ("930c31948dac5c3547324e2c511b04e61791637c25a66e0cb1d61bfc525b4fe9", "9563e71cb448a17e4514306de879d86ca4de3b15bfbab9ce86170719b17140f1", "a1ad848d0ab26566aefefe07fc7ff9c514ae70c8ad6cca9c4aea2ad3e9d39056", "preserving", [], "changes only the fixed vertical translation/flip inference mixture"),
"8dc9258e4d7db10cf57a518c9a5a4b842cfc3d421d882698f0a4be19b68e06cd": ("61d31cee5a153b46f784c609fa799f18bad71efe65c1028b3e4f25014f70e385", "9563e71cb448a17e4514306de879d86ca4de3b15bfbab9ce86170719b17140f1", "a1ad848d0ab26566aefefe07fc7ff9c514ae70c8ad6cca9c4aea2ad3e9d39056", "preserving", [], "uses the learned 1x1 view-fusion result as a fixed additive residual"),
"aa861c7f2581337dff6b68a921f454c7a77e3fe3b5737f7b3333a2282f55f68c": ("7260b7e5b371cf204af29fa298b311d7af51bf7266b51e995a7f48cb00104fb5", "9563e71cb448a17e4514306de879d86ca4de3b15bfbab9ce86170719b17140f1", "a1ad848d0ab26566aefefe07fc7ff9c514ae70c8ad6cca9c4aea2ad3e9d39056", "changing", ["routing", "aggregation", "channel_interaction"], "adds a mean-and-variance descriptor driving a tanh channel gate"),
"ef1c61ff7781454794dbe3c4c92e0625d27d9803c51574c250a58a7cbd0d2617": ("ed930a1c6fcbd081620421437a98279fb9984ad76e608b92a0ceb59451bddd89", "9563e71cb448a17e4514306de879d86ca4de3b15bfbab9ce86170719b17140f1", "a1ad848d0ab26566aefefe07fc7ff9c514ae70c8ad6cca9c4aea2ad3e9d39056", "preserving", [], "adds a fixed local convolutional residual refinement"),
"0582e944d04f821c8b4df7b55de45e00340f1d14fa0cd41465124d3d7d8167af": ("a5167208f89f90d93d3014f2290c69a92c633022a669f450e3d9d2bffe2637ec", "a931485e758953f60c43be8f3e17c74ec23e126c8ac3ae79dda5fe1841d2da43", "cebab4c4aaa6e08136f894b855cf72a9f2f385bc5250eef5856067c31df85ab8", "preserving", [], "adds a fixed local convolutional residual refinement and classifier EMA setting"),
"d137f173ce6433d23ceb7f44d65a2d973d42a09da34c1d032d168ddddf7bbc37": ("364d9fca78dc1fe99d71da1517f7dea786523e74c28d19584bc581a7c061f0f9", "a931485e758953f60c43be8f3e17c74ec23e126c8ac3ae79dda5fe1841d2da43", "cebab4c4aaa6e08136f894b855cf72a9f2f385bc5250eef5856067c31df85ab8", "preserving", [], "adds a training-only paired Dropout consistency objective without recurrent feature state"),
}

PROPOSALS = {
    sha: proposal for sha, proposal in zip(SPECS, (50, 51, 91, 120, 124, 126, 130, 136, 192, 196))
}

BASE = {
"input_units": "grayscale image pixels", "input_transform": "identity image tensor input", "embedding": "not applicable: continuous-valued grayscale pixels enter convolutional operators directly", "position": "implicit local image neighborhoods; no explicit positional embedding", "mixing": "dense Conv2d backbone with learned 1x1 paired-view fusion", "routing": "fixed original/horizontal-flip feature pairing and tensor concatenation", "state": "no persistent recurrent feature state", "feedforward": "no separate transformer feedforward block", "parameter_construction": "free learned convolutional and affine parameters", "sharing": "no explicit parameter tying beyond source-defined module reuse", "normalization": "BatchNorm2d feature normalization", "connectivity": "fixed convolutional residual stages followed by paired-view fusion and affine classification", "aggregation": "flattened fused spatial grid", "output": "learned affine ten-class logits", "symmetry": "fixed horizontal-flip paired feature construction; no equivariant parameter construction", "conditional_compute": "fixed architecture; no input-dependent early exit or experts", "activation": "GELU", "stochasticity": "Dropout in the affine classifier during training", "bottleneck": "no separate factorized bottleneck", "iteration": "fixed feedforward stages", "other": "complete exact retained parent and child train.py sources were directly read and diffed", "spatial_units": "image grid", "spatial_operator": "dense local Conv2d", "scale_representation": "two MaxPool2d reductions followed by a fixed-resolution convolutional grid", "spatial_readout": "flattened spatial grid through an affine classifier", "channel_interaction": "learned 1x1 paired-view fusion convolution", "spatial_downsampling": "two MaxPool2d(2) operations"}

OVERRIDES = {
"d2016695edf7": {"mixing": "dense Conv2d plus local depthwise near/wide context branches and 1x1 mixing", "connectivity": "paired-view fusion feeds parallel local/dilated depthwise context branches concatenated before the affine classifier", "aggregation": "flattened locally contextualized spatial grid", "spatial_operator": "dense local Conv2d plus depthwise 3x3 and dilated 3x3 local Conv2d", "channel_interaction": "learned 1x1 paired-view fusion and 192-to-96 1x1 context mixing"},
"097a2f73ec4f": {"aggregation": "flattened spatial grid plus learned adaptive-average-pooled 4x4 skip logits", "scale_representation": "two MaxPool2d reductions plus an AdaptiveAvgPool2d(4,4) skip grid", "spatial_readout": "affine flattened-grid logits summed with adaptive-average-pooled affine skip logits"},
"19945e1c3924": {"mixing": "dense Conv2d backbone, learned 1x1 paired-view fusion, and MLP channel gate", "routing": "global average of paired invariant/disagreement channels drives a tanh channel scale", "aggregation": "global average paired-view descriptor and flattened gated spatial grid", "connectivity": "paired-view fusion followed by multiplicative tanh channel gating and affine classification", "activation": "GELU and tanh channel gate", "channel_interaction": "128-to-16-to-64 affine paired-view channel gate"},
"d05ae9c873b3": {"mixing": "dense Conv2d backbone plus learned 1x1 bilinear left/right projections", "routing": "fixed bilinear outer-product statistic of paired-view spatial features", "aggregation": "flattened spatial grid and signed square-root normalized bilinear channel statistic", "connectivity": "paired-view fusion feeds both affine spatial logits and a bilinear statistic logit head", "spatial_readout": "affine flattened-grid logits summed with normalized bilinear-statistic logits", "channel_interaction": "two learned 64-to-16 1x1 projections combined by a bilinear matrix product"},
"fba2269f217c": {"routing": "fixed original, one-pixel-up, and one-pixel-down views with horizontal flips and fixed logit weights", "symmetry": "fixed horizontal-flip and vertical-translation inference ensemble; no equivariant parameter construction"},
"8dc9258e4d7d": {"connectivity": "paired-view invariant feature grid plus a learned 1x1 fusion residual before affine classification"},
"aa861c7f2581": {"mixing": "dense Conv2d backbone plus a 128-to-16-to-64 affine channel-gate MLP", "routing": "channel mean and variance drive a tanh channel scale before the final pool", "aggregation": "global mean-and-variance channel descriptor and flattened gated spatial grid", "connectivity": "convolutional residual stages followed by multiplicative channel gating, paired-view fusion, and affine classification", "activation": "GELU and tanh channel gate", "channel_interaction": "128-to-16-to-64 affine mean-and-variance channel gate"},
"ef1c61ff7781": {"mixing": "dense Conv2d backbone plus fixed local 1x1-3x3-1x1 residual refinement", "connectivity": "paired-view fusion followed by local convolutional residual refinement and affine classification", "spatial_operator": "dense local Conv2d including a 1x1-3x3-1x1 refinement block"},
"0582e944d04f": {"mixing": "dense Conv2d backbone plus fixed local 1x1-3x3-1x1 residual refinement", "connectivity": "paired-view fusion followed by local convolutional residual refinement and affine classification", "spatial_operator": "dense local Conv2d including a 1x1-3x3-1x1 refinement block"},
"d137f173ce64": {"state": "training-only paired Dropout logits are stored for the current loss call; no persistent recurrent feature state", "other": "complete exact retained parent and child train.py sources were directly read and diffed; the paired Dropout consistency calculation is training-only"},
}

def _profile(source_sha256):
    if source_sha256 not in SPECS:
        raise KeyError("unreviewed Fashion source: " + source_sha256)
    fp = dict(BASE); fp.update(OVERRIDES[source_sha256[:12]])
    return fp

def profiles():
    candidates = ROOT / RUN / "candidates"
    result = []
    for sha, (child_id, parent_sha, parent_id, classification, changed, reason) in SPECS.items():
        child, parent = source(candidates / child_id), source(candidates / parent_id)
        if digest(child) != sha or digest(parent) != parent_sha:
            raise ValueError("source binding changed for " + child_id)
        fp = _profile(sha)
        missing = [key for key in CORE + TASK_KEYS["fashion"].split() if not fp.get(key)]
        if missing: raise ValueError("incomplete fingerprint: " + repr(missing))
        diff = list(difflib.unified_diff(parent["train.py"].splitlines(), child["train.py"].splitlines(), lineterm=""))
        family = {"task": "fashion", "input": "grayscale image grid", "mixing": "local convolution", "routing": "fixed paired-view local tensor flow", "readout": "flattened spatial affine classifier"}
        if sha.startswith("097a"): family["readout"] = "flattened spatial affine plus adaptive-average-pooled skip logits"
        if sha.startswith(("1994", "aa86")): family["routing"] = "input-dependent global-statistic channel gating"
        if sha.startswith("d05a"): family["readout"] = "flattened spatial affine plus bilinear channel-statistic logits"
        if sha.startswith("d201"): family["mixing"] = "local dense/depthwise convolution"
        training = {"optimizer": "AdamW (declared)", "objective": "cross entropy with label smoothing (declared)", "augmentation": "random horizontal flip (declared)"}
        if sha.startswith("d137"): training["objective"] = "cross entropy with label smoothing plus symmetric paired-Dropout KL consistency (declared)"
        inference = {"procedure": "fixed original/horizontal-flip paired feature fusion and positive scalar logit calibration"}
        if sha.startswith("fba2"): inference = {"procedure": "fixed original/up/down vertical translations and horizontal flips use fixed 0.50/0.25/0.25 logit weights"}
        result.append({"source_sha256": sha, "reference_run": RUN, "reference_proposal": PROPOSALS[sha], "candidate_id": child_id, "parent_candidate_id": parent_id, "parent_source_sha256": parent_sha, "transition_classification": classification, "changed_components": changed, "fingerprint": fp, "family_signature": family, "family_label": "paired-view local CNN", "training": training, "inference": inference, "notes": "Complete exact queue-bound parent and child train.py sources were directly read and diffed. " + reason + ".", "evidence": [{"kind": "complete exact-program queue-bound parent-child diff review", "source_sha256": sha, "parent_source_sha256": parent_sha, "candidate_id": child_id, "retained_parent_candidate_id": parent_id, "directed_parent_source_sha256": parent_sha, "classification": classification, "changed_components": changed, "reason": reason, "changed_lines": {"added": sum(x.startswith("+") and not x.startswith("+++") for x in diff), "removed": sum(x.startswith("-") and not x.startswith("---") for x in diff)}}]})
    return result

