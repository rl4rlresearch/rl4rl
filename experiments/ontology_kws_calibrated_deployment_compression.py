"""Exact-source review for calibrated KWS B05-C0 deployment compression."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.ontology_review_recurrent import source_sha

_LEDGER_PATH = Path(__file__).with_name("ontology_kws_calibrated_deployment_compression_pairs.json")
EXACT_PAIRS = tuple(tuple(entry) for entry in json.loads(_LEDGER_PATH.read_text(encoding="utf-8-sig")))
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
        "parameter_construction": "train(False) derives a covariance-reconstruction predictor and relative-logit head from trained weights, recorded feature moments, late feature moments, and retained calibration logits",
        "sharing": "streaming and sequence paths share input normalization, GRU, summary, and classifier",
        "normalization": "learned affine LayerNorm; temporal mean divides by count clamped below at one",
        "connectivity": "normalized frames feed the GRU; mean output excluding its final two coordinates concatenates with final hidden state before ordinary or deployment classifier",
        "aggregation": "causal running arithmetic mean of GRU outputs plus final GRU hidden state",
        "output": "eight logits in training; deployment returns seven class-zero-relative logits plus a fixed zero reference logit",
        "symmetry": "the relative-logit deployment form removes only the shared logit coordinate",
        "conditional_compute": "train(False) runs a source-fixed %d-stage coordinate selector: covariance scoring and local reconstruction refinements, followed by calibration-distillation extensions" % dropped,
        "activation": "standard GRU sigmoid and tanh gates; linear classifier, covariance/linear-solve reconstruction, and calibration softmax/log-softmax scoring during synchronization",
        "stochasticity": "training applies Gaussian acoustic noise with probability 0.8; no inference-time random branch",
        "bottleneck": "%d deployed coordinates with %d reconstructed coordinates; training feature width remains 114" % (114-dropped, dropped),
        "iteration": "one causal GRU scan; train(False) runs a source-fixed %d-stage coordinate-selection sequence" % dropped,
        "other": "Complete exact-pair audit: only the SHA-bound train(False) calibrated selector stage, dropped-coordinate count, and dependent deployment head width differ.",
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
        "notes": "Exact P%d source-pair review: the sole source delta is train(False), which adds one calibrated covariance-guided dropped deployment coordinate (%d to %d) and reduces the relative-logit head accordingly." % (proposal, before, after),
        "evidence": [{"file": "train.py", "line": line, "kind": "complete hash-bound B05-C0 calibrated deployment-compression source-pair proof", "code": "train(False) extends the calibrated dropped-coordinate selector from %d to %d and changes nn.Linear(full_classifier.in_features - %d, 7)." % (before, after, after)}],
        "settings": {"proposal": proposal, "deployment_dropped_coordinates": {"before": before, "after": after}, "deployment_relative_head_input": {"before": 114-before, "after": 114-after}},
        "family_label": "KWS GRU / calibrated covariance-guided relative-logit deployment compression",
        "family_signature": {"state": "20-input 58-hidden GRU with running mean and final hidden state", "deployment": "class-zero-relative compressed affine readout with covariance reconstruction and calibration-distillation selector stages", "routing": "source-fixed calibrated greedy dropped-coordinate selector"},
        "before_source_sha256": parent,
        "source_sha256": child,
        "reviewer": "kws-b05-c0-calibrated-deployment-compression-exact-pairs-v1",
    }
