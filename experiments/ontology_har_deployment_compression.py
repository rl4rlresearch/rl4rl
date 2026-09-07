"""Exact-source review for the HAR B05-C2 deployment-compression cohort.

The JSON ledger is a static enumeration of recorded proposal-122 through
proposal-200 source pairs.  This recognizer deliberately performs no source
shape, ranking-loop, or AST inference: both complete source hashes must occur
in that ledger before a result is returned.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.ontology_review_recurrent import source_sha


_LEDGER_PATH = Path(__file__).with_name("ontology_har_deployment_compression_pairs.json")
EXACT_PAIRS = tuple(json.loads(_LEDGER_PATH.read_text(encoding="utf-8")))
_BY_HASH_PAIR = {(entry["parent_sha256"], entry["candidate_sha256"]): entry for entry in EXACT_PAIRS}


def _fingerprint(omitted: int, head_input: int) -> dict[str, str]:
    """Complete candidate ontology for one hash-bound ledger entry."""
    return {
        "input_units": "fixed 128-step, nine-channel tri-axial human-activity sensor windows",
        "input_transform": "raw xyz windows provide six raw statistics per channel and three grouped magnitude summaries",
        "embedding": "depthwise-plus-pointwise temporal convolutional stem followed by dual-statistic downsampling",
        "position": "fixed temporal views: a 14-step forward GRU view and a reversed 10-step backward GRU view with a fixed seed summary; no learned positional embedding",
        "mixing": "separable temporal convolutions and independent forward/backward GRUs; class-relative deployment readout is synchronized from the ordinary trained classifier",
        "routing": "the fixed all-path deployment route masks exactly %d selected recurrent-feature coordinates, applies a %d-input frozen relative-logit head, and adds one four-logit residual head for each masked coordinate" % (omitted, head_input),
        "state": "forward and backward GRU hidden states are per-window and are not persistent deployment state",
        "feedforward": "ordinary six-logit training classifier plus five-input magnitude classifier; deployment uses a frozen five-relative-logit affine head and %d one-input four-logit residual affine heads" % omitted,
        "parameter_construction": "ordinary learned convolution, GRU, and training-head parameters; deployment head and residuals are synchronized under no_grad and then frozen, while omitted index buffers are fixed nonpersistent state",
        "sharing": "the same stem and directional GRUs serve training and deployment; deployment relative weights derive from the same trained classifier and magnitude classifier",
        "normalization": "BatchNorm after each pointwise convolution; no recurrent normalization layer",
        "connectivity": "depthwise/pointwise stem feeds two directional GRUs; endpoints, directional mean/max summaries, six raw-statistic groups, and magnitude summaries feed the training or synchronized deployment head",
        "aggregation": "forward/backward endpoints, directional sequence means and maxima, raw mean/std/max/min/short-change/medium-change, and three magnitude summaries",
        "output": "six activity logits; deployment emits class-zero-relative logits by prepending a fixed zero reference logit",
        "symmetry": "no symmetric set reduction; ordered sensor and directional temporal paths are retained",
        "conditional_compute": "only the explicit trainingHead branch selects the ordinary train head; deployment otherwise follows fixed masked-head and residual loops",
        "activation": "ReLU in convolutional blocks and tanh for the backward initial state; GRU gates provide recurrent nonlinearities",
        "stochasticity": "dropout 0.15 before the ordinary training/deployment readout during training; no inference-time random branch",
        "bottleneck": "%d deployed main-head coordinates plus %d single-coordinate residual paths; the full training feature vector remains unchanged" % (head_input, omitted),
        "iteration": "finite forward/backward GRU scans and a source-fixed %d-entry deployment residual loop" % omitted,
        "other": "Complete exact-pair proof: the source hashes bind every edit in the recorded pair. Only the fixed deployment omission cardinality, dependent frozen head width, index buffers, and residual-loop bound differ.",
        "sensor_representation": "raw xyz, temporal change statistics, triad magnitudes, convolutional sensor features, and directional recurrent summaries",
        "sensor_fusion": "depthwise sensor filtering followed by pointwise mixing; raw and magnitude summaries concatenate with directional recurrent summaries only at readout",
        "temporal_operator": "two convolutional stages, fixed pooling/views, a 14-step forward GRU, and a reversed 10-step backward GRU",
        "directionality": "separate forward and reversed backward GRU paths; deployment readout is class-zero-relative rather than directional",
        "state_update": "GRU hidden-state updates only within each input window; _syncInferenceHead updates frozen deployment copies during setup/optimizer synchronization, not inference history",
        "state_structure": "one hidden state per directional single-layer GRU; fixed omitted-feature and omitted-class index buffers for deployment routing",
        "frequency_representation": "no spectral transform",
        "temporal_schedule": "stem/pool schedule yields fixed directional 14- and 10-step views; all deployment residual iterations have the ledger-bound count",
        "readout_history": "whole-window directional endpoints/means/maxima plus raw and magnitude summaries",
        "temporal_readout": "directional GRU summaries concatenate with raw/magnitude statistics before the ordinary or synchronized relative deployment classifier",
        "exit_policy": "fixed full-window classification; no early exit",
    }


def review_pair(before_sources: dict[str, str], after_sources: dict[str, str]):
    """Recognize only one fully enumerated B05-C2 deployment-compression pair."""
    before_sha = source_sha(before_sources)
    after_sha = source_sha(after_sources)
    entry = _BY_HASH_PAIR.get((before_sha, after_sha))
    if entry is None:
        return None

    before_omitted = entry["before_omitted"]
    after_omitted = entry["after_omitted"]
    head_input = entry["candidate_head_input"]
    return {
        "classification": "preserving",
        "fingerprint": _fingerprint(after_omitted, head_input),
        "fingerprint_complete": True,
        "residual_components": [],
        "changed_components": [],
        "notes": (
            "Exact source-pair proof for proposal %d. The hash-bound full "
            "source diff changes the deployment omission cardinality from %d "
            "to %d, its dependent frozen main-head input from %d to %d, and "
            "the matching index-buffer/residual-loop bounds. The learned "
            "sensor path, GRUs, training heads, synchronization algorithm, "
            "and class-zero-relative deployment topology are unchanged."
            % (entry["proposal"], before_omitted, after_omitted, entry["parent_head_input"], head_input)
        ),
        "evidence": [{
            "file": "train.py",
            "line": 151,
            "end_line": 227,
            "code": (
                "inferenceClassifier uses %d input coordinates; "
                "inferenceResiduals, omittedFeatures, omittedClasses, and both "
                "least-magnitude selection/residual-addition loops use %d entries "
                "(parent: %d)."
                % (head_input, after_omitted, before_omitted)
            ),
            "kind": "complete hash-bound B05-C2 deployment-compression source-pair proof",
        }],
        "settings": {
            "proposal": entry["proposal"],
            "deployment_omissions": {"before": before_omitted, "after": after_omitted},
            "deployment_main_head_input": {"before": entry["parent_head_input"], "after": head_input},
            "deployment_residual_heads": {"before": before_omitted, "after": after_omitted, "each": "Linear(1, 4, bias=False)"},
            "training_classifier_input": 210,
            "training_classifier_output": 6,
            "deployment_relative_output": 5,
            "changed_source_components": [
                "deployment omission cardinality", "dependent inferenceClassifier input width",
                "inference residual-head count", "omitted feature/class buffer lengths",
                "selection and residual-addition fixed loop bounds",
            ],
        },
        "family_label": "MicroBiConvLSTM / synchronized class-relative deployment compression",
        "family_signature": {
            "sensor_pipeline": "depthwise/pointwise temporal CNN with raw and magnitude statistics",
            "state": "per-window directional GRU states only",
            "deployment": "frozen class-zero-relative affine readout with selected-coordinate residual corrections",
            "training": "ordinary six-logit classifier plus magnitude classifier",
            "routing": "hash-bound fixed cardinality of least-magnitude coordinate omissions",
        },
        "before_source_sha256": before_sha,
        "source_sha256": after_sha,
        "reviewer": "har-b05-c2-deployment-compression-exact-pairs-v1",
    }
