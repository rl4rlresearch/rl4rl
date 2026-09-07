"""Exact-source review for KWS B05-C0 progressive deployment compression.

This recognizer is intentionally a seven-pair SHA ledger.  It does not
generalize from covariance selection, classifier widths, or train hooks.
"""
from __future__ import annotations

from experiments.ontology_review_recurrent import source_sha

EXACT_PAIRS = (
    (60, 2, 3, "e64e6f7717e5ea224240fdb269bed42d1f14724afeb449eef4e42fc8f36bdde7", "d6f193af177c73c52db4c39032e7942866c965346a96e746130d0e5ab4397a02"),
    (61, 3, 4, "d6f193af177c73c52db4c39032e7942866c965346a96e746130d0e5ab4397a02", "9b5f84c0d163f9859b921086b858aff4ccec1032c8f2f290d47c26714cbe3b2d"),
    (62, 4, 5, "9b5f84c0d163f9859b921086b858aff4ccec1032c8f2f290d47c26714cbe3b2d", "7582c84a9ec6313ec9f7c0b1ffe1cf714c56b995795622f185d540ed988b4c80"),
    (63, 5, 6, "7582c84a9ec6313ec9f7c0b1ffe1cf714c56b995795622f185d540ed988b4c80", "d73364b2c32f3ad636d7ff28f38c1abbe88907f8bd0ba168dda1683ad22933bd"),
    (64, 6, 7, "d73364b2c32f3ad636d7ff28f38c1abbe88907f8bd0ba168dda1683ad22933bd", "432808499e9219b99534e6a45863987d3711cff9b2d860a4c9c5dd1a874a31eb"),
    (65, 7, 8, "432808499e9219b99534e6a45863987d3711cff9b2d860a4c9c5dd1a874a31eb", "b7e0e2cd3d16baccd7727dccffc05f5f1355df0259f49a257ae132fd56ee711d"),
    (66, 8, 9, "b7e0e2cd3d16baccd7727dccffc05f5f1355df0259f49a257ae132fd56ee711d", "68e63d068ab5e8133b3b50a0848cef963c6db1f03a80f2f0d0f0a5033917c521"),
)
_BY_HASH = {(parent, child): (proposal, before, after) for proposal, before, after, parent, child in EXACT_PAIRS}


def _fingerprint(dropped: int) -> dict[str, str]:
    return {
        "input_units": "frozen 20-band acoustic feature frames",
        "input_transform": "learned affine LayerNorm over each acoustic frame",
        "embedding": "continuous acoustic features feed the GRU directly",
        "position": "causal streaming frame order",
        "mixing": "one standard unidirectional 20-input, 58-hidden GRU",
        "routing": "fixed recurrent path; deployment selects 114 mean/final-state coordinates after train(False) synchronization",
        "state": "GRU hidden state, running output sum, and scalar frame count",
        "feedforward": "ordinary 114-to-8 affine classifier during training; deployment uses a 114-minus-%d-to-7 relative-logit affine classifier" % dropped,
        "parameter_construction": "train(False) derives a covariance-guided reduced relative-logit classifier and fixed predictor buffers from the trained classifier and recorded feature moments",
        "sharing": "streaming and sequence paths share input normalization, GRU, summary, and classifier",
        "normalization": "learned affine LayerNorm; temporal mean divides by count clamped below at one",
        "connectivity": "normalized frames feed the GRU; mean output excluding its final two coordinates concatenates with final hidden state before ordinary or deployment classifier",
        "aggregation": "causal running arithmetic mean of GRU outputs plus final GRU hidden state",
        "output": "eight logits in training; deployment returns seven class-zero-relative logits plus a fixed zero reference logit",
        "symmetry": "the relative-logit deployment form removes only the shared logit coordinate",
        "conditional_compute": "train(False) selects %d coordinates with covariance-guided greedy scores and runs the reduced deployment head" % dropped,
        "activation": "standard GRU sigmoid and tanh gates; linear classifier and covariance/linear-solve deployment synchronization",
        "stochasticity": "training applies Gaussian acoustic noise with probability 0.8; no inference-time random branch",
        "bottleneck": "%d deployed coordinates with %d reconstructed coordinates; training feature width remains 114" % (114 - dropped, dropped),
        "iteration": "one causal GRU scan; train(False) runs a source-fixed %d-stage coordinate-selection sequence" % dropped,
        "other": "Complete exact-pair audit: only the SHA-bound train(False) selector stage, dropped-coordinate count, and dependent deployment head width differ.",
        "acoustic_representation": "protected log-mel acoustic frames",
        "state_update": "standard GRU update with output-sum and count accumulation",
        "state_structure": "hidden [batch,1,58], running output sum [batch,58], and count [batch,1]",
        "temporal_schedule": "fixed source schedule: retain a cropped frame window and every other final three frames",
        "readout_history": "mean GRU output without its final two coordinates concatenated with final hidden state",
        "exit_policy": "fixed horizon; no early-exit hook",
    }


def review_pair(before_sources: dict[str, str], after_sources: dict[str, str]):
    parent, child = source_sha(before_sources), source_sha(after_sources)
    entry = _BY_HASH.get((parent, child))
    if entry is None:
        return None
    proposal, before, after = entry
    source = after_sources.get("train.py", "").splitlines()
    line = next((i for i, text in enumerate(source, 1) if "full_classifier.in_features - %d" % after in text), 1)
    return {
        "classification": "changing",
        "fingerprint": _fingerprint(after),
        "fingerprint_complete": True,
        "residual_components": [],
        "changed_components": ["output", "parameter_construction", "routing", "bottleneck"],
        "notes": "Exact P%d source-pair review: train(False) adds one covariance-guided dropped deployment coordinate (%d to %d) and reduces the relative-logit head accordingly. The recurrent state and acoustic path are retained, but deployment output construction changes." % (proposal, before, after),
        "evidence": [{"file": "train.py", "line": line, "kind": "complete hash-bound B05-C0 progressive deployment-compression source-pair proof", "code": "train(False) extends the covariance-guided dropped-coordinate selector from %d to %d and changes nn.Linear(full_classifier.in_features - %d, 7)." % (before, after, after)}],
        "settings": {"proposal": proposal, "deployment_dropped_coordinates": {"before": before, "after": after}, "deployment_relative_head_input": {"before": 114-before, "after": 114-after}},
        "family_label": "KWS GRU / covariance-guided relative-logit deployment compression",
        "family_signature": {"state": "20-input 58-hidden GRU with running mean and final hidden state", "deployment": "class-zero-relative compressed affine readout with covariance-guided reconstructed coordinates", "routing": "source-fixed greedy dropped-coordinate selector"},
        "before_source_sha256": parent,
        "source_sha256": child,
        "reviewer": "kws-b05-c0-progressive-deployment-compression-exact-pairs-v1",
    }

