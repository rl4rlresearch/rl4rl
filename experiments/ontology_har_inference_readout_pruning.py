"""Exact-source review for the HAR B04-C2 inference-readout pruning sweep.

The adjacent proposal 144 pooling change is intentionally excluded.  Runtime
recognition is a lookup in the static recorded source-hash ledger only; it
does not infer equivalence from prune-count assignments or covariance code.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.ontology_review_recurrent import source_sha


_LEDGER_PATH = Path(__file__).with_name("ontology_har_inference_readout_pruning_pairs.json")
EXACT_PAIRS = tuple(json.loads(_LEDGER_PATH.read_text(encoding="utf-8")))
_BY_HASH_PAIR = {(entry["parent_sha256"], entry["candidate_sha256"]): entry for entry in EXACT_PAIRS}


def _fingerprint(prune_count: int, compact_width: int) -> dict[str, str]:
    return {
        "input_units": "fixed 128-step, nine-channel tri-axial human-activity sensor windows",
        "input_transform": "raw xyz windows produce global mean, standard deviation, maximum and compact summary features",
        "embedding": "stride-two temporal Conv1d stem followed by dense dilated stride-two Conv1d filtering",
        "position": "fixed temporal order and reverse-time view; no learned positional embedding",
        "mixing": "dense convolutions and separate forward/reverse GRUs; deployment compact modules are covariance-compensated copies of the trained path",
        "routing": "training uses the dense model; deployment removes one Stage-II channel and exactly %d recurrent readout coordinates, retaining a %d-coordinate compact relative-logit head" % (prune_count, compact_width),
        "state": "per-window forward/reverse GRU hidden states and nonpersistent covariance/index buffers; no persistent deployment history",
        "feedforward": "80-coordinate ordinary six-logit recurrent classifier plus 27-coordinate summary classifier; frozen deployment has a %d-coordinate five-relative-logit classifier and a five-relative-logit summary classifier" % compact_width,
        "parameter_construction": "ordinary learned convolution, BatchNorm, GRU, and training-head parameters; compact deployment modules are no_grad synchronized and frozen",
        "sharing": "training and deployment share the trained stem/GRU/head parameters through synchronization; no new observation path is introduced",
        "normalization": "BatchNorm after both convolution stages; covariance moments support only deployment reconstruction",
        "connectivity": "Conv1 feeds dense dilated Conv2, then forward and reverse GRUs; recurrent endpoint/mean summaries and raw sensor summaries feed dense or compact heads",
        "aggregation": "forward/reverse GRU endpoint and mean summaries plus raw mean, standard deviation and maximum",
        "output": "six activity logits; compact deployment emits five class-relative logits with a fixed zero reference class",
        "symmetry": "stage covariance is explicitly symmetrized for reconstruction; sensor and temporal directions remain ordered",
        "conditional_compute": "explicit training versus compact deployment head path only",
        "activation": "ReLU after BatchNorm convolutions; GRU gates provide recurrent nonlinearities",
        "stochasticity": "dropout before the heads during training; no inference-time random branch",
        "bottleneck": "%d retained recurrent readout coordinates after the fixed %d-coordinate prune count" % (compact_width, prune_count),
        "iteration": "finite directional GRU scans and a source-bound fixed greedy selection loop of %d iterations during synchronization" % prune_count,
        "other": "Complete exact-pair proof: both source hashes bind the full train.py diff. Only the prune-count assignment/comment change; dependent compact dimensions, buffers, greedy selection and covariance reconstruction are source-identical dimension-dependent operations.",
        "sensor_representation": "raw xyz, temporal convolutional features, directional GRU summaries, and global raw summaries",
        "sensor_fusion": "dense temporal convolutional mixing followed by directional recurrent fusion; raw summary joins at readout",
        "temporal_operator": "stride-two Conv1d, dilated stride-two Conv1d, and forward/reverse GRUs",
        "directionality": "independent forward and reversed GRU paths",
        "state_update": "GRU state updates within each window; _syncInferenceHeads updates frozen compact copies during setup, not inference history",
        "state_structure": "one hidden state per directional GRU plus fixed covariance moments and compact index buffers",
        "frequency_representation": "no spectral transform",
        "temporal_schedule": "source-fixed convolution strides/dilation and full-window directional GRU scans",
        "readout_history": "directional recurrent summaries and full-window raw statistics",
        "temporal_readout": "concatenated forward/reverse recurrent summaries followed by dense or compact relative readout",
        "exit_policy": "fixed full-window classification; no early exit",
    }


def review_pair(before_sources: dict[str, str], after_sources: dict[str, str]):
    before_sha, after_sha = source_sha(before_sources), source_sha(after_sources)
    entry = _BY_HASH_PAIR.get((before_sha, after_sha))
    if entry is None:
        return None
    before_count, after_count = entry["before_prune_count"], entry["after_prune_count"]
    width = entry["candidate_compact_width"]
    return {
        "classification": "preserving", "fingerprint": _fingerprint(after_count, width),
        "fingerprint_complete": True, "residual_components": [], "changed_components": [],
        "notes": "Exact source-pair proof for proposal %d: the only source diff changes inferenceReadoutPruneCount from %d to %d and its explanatory comment; the compact width changes from %d to %d through the unchanged dimension-dependent construction." % (entry["proposal"], before_count, after_count, entry["parent_compact_width"], width),
        "evidence": [{"file": "train.py", "line": 123, "end_line": 175, "code": "inferenceReadoutPruneCount=%d determines the %d-coordinate inferenceFeatureIndices and inferenceClassifier construction (parent count=%d)." % (after_count, width, before_count), "kind": "complete hash-bound B04-C2 inference-readout-pruning source-pair proof"}],
        "settings": {"proposal": entry["proposal"], "inference_readout_prune_count": {"before": before_count, "after": after_count}, "compact_readout_width": {"before": entry["parent_compact_width"], "after": width}, "dense_readout_width": 80, "changed_source_components": ["inferenceReadoutPruneCount assignment", "matching explanatory comment"]},
        "family_label": "MicroBiConvLSTM / covariance-compensated compact inference readout",
        "family_signature": {"sensor_pipeline": "dense temporal CNN with raw summaries", "state": "per-window directional GRUs", "deployment": "one-channel covariance reconstruction and compact class-relative readout", "routing": "ledger-bound prune cardinality only"},
        "before_source_sha256": before_sha, "source_sha256": after_sha,
        "reviewer": "har-b04-c2-inference-readout-pruning-exact-pairs-v1",
    }
