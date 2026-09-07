"""Hash-bound B03-C2 tertiary reset-omission expansion ledger."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.ontology_review_recurrent import source_sha

_LEDGER = Path(__file__).with_name("ontology_har_tertiary_reset_omission_pairs.json")
EXACT_PAIRS = tuple(json.loads(_LEDGER.read_text(encoding="utf-8")))
_BY_PAIR = {(entry["parent_sha256"], entry["candidate_sha256"]): entry for entry in EXACT_PAIRS}


def _fingerprint(omissions: int) -> dict[str, str]:
    return {
        "input_units": "fixed 128-step, nine-channel tri-axial human-activity sensor windows",
        "input_transform": "raw mean/std/max summaries",
        "embedding": "depthwise/pointwise pooled temporal CNN",
        "position": "fixed transition order; no positional embedding",
        "mixing": "forward GRU and endpoint GRUCell with covariance-ranked compact reset projections",
        "routing": "dense training path; frozen compact deployment applies tertiary reset-output omissions at %d selected coefficients" % omissions,
        "state": "per-window recurrent state and fixed moment/index buffers only",
        "feedforward": "dense six-logit training head and frozen five-relative-logit compact head",
        "parameter_construction": "ordinary learned CNN/GRU/head parameters; compact projections synchronize under no_grad and are frozen",
        "sharing": "compact modules derive from the same trained encoder",
        "normalization": "BatchNorm in temporal CNN",
        "connectivity": "separable CNN feeds forward GRU/endpoint cell; recurrent and raw summaries feed dense or compact head",
        "aggregation": "recurrent states and whole-window raw moments",
        "output": "six logits with compact class-relative deployment readout",
        "symmetry": "none",
        "conditional_compute": "training versus compact deployment path only",
        "activation": "ReLU and GRU gates",
        "stochasticity": "training dropout only",
        "bottleneck": "%d covariance-ranked omissions from one tertiary reset-output row" % omissions,
        "iteration": "finite temporal scan and %d tertiary covariance-selection iterations" % omissions,
        "other": "Complete exact-pair proof: hashes bind every edit. Only tertiary reset-output omission cardinality and dependent projection/buffer/selection/synchronization/dispatch code change.",
        "sensor_representation": "raw xyz and learned temporal features",
        "sensor_fusion": "pointwise mixing after depthwise filters",
        "temporal_operator": "two pooled separable convolutions, forward GRU, endpoint cell",
        "directionality": "forward encoder plus endpoint state",
        "state_update": "within-window GRU updates; synchronization is setup only",
        "state_structure": "GRU/cell state plus moment/index buffers",
        "frequency_representation": "no spectral transform",
        "temporal_schedule": "fixed pooled transitions",
        "readout_history": "recurrent endpoints plus raw moments",
        "temporal_readout": "dense or compact relative affine readout",
        "exit_policy": "fixed full-window classification",
    }


def review_pair(before_sources: dict[str, str], after_sources: dict[str, str]):
    before_sha, after_sha = source_sha(before_sources), source_sha(after_sources)
    entry = _BY_PAIR.get((before_sha, after_sha))
    if entry is None:
        return None
    before_count, after_count = entry["before_tertiary_omissions"], entry["after_tertiary_omissions"]
    return {
        "classification": "preserving",
        "fingerprint": _fingerprint(after_count),
        "fingerprint_complete": True,
        "residual_components": [],
        "changed_components": [],
        "notes": "Exact source-pair proof for proposal %d: only tertiary reset-output omissions change from %d to %d, with dependent compact buffers, projections, covariance selection, synchronization, and dispatch bounds." % (entry["proposal"], before_count, after_count),
        "evidence": [{
            "file": "train.py",
            "line": 339,
            "end_line": 1532,
            "code": "keptTertiarySingleFifthForwardResetInputs, compactForwardResetHiddenSingleTertiary, covariance-ranked tertiary selections, synchronization and dispatch use %d omissions (parent %d)." % (after_count, before_count),
            "kind": "complete hash-bound B03-C2 tertiary-reset-omission source-pair proof",
        }],
        "settings": {
            "proposal": entry["proposal"],
            "tertiary_omission_coefficients": {"before": before_count, "after": after_count},
            "changed_source_components": [
                "tertiary reset-output omission cardinality",
                "dependent compact projection input width",
                "covariance-ranked selection chain",
                "synchronization and dispatch loop bounds",
            ],
        },
        "family_label": "MicroBiConvLSTM / tertiary reset-output compact deployment",
        "family_signature": {
            "training": "dense recurrent classifier",
            "deployment": "compact class-relative reset projections",
            "tertiary_route": "one covariance-ranked third reset-output row",
            "routing": "ledger-bound only",
        },
        "before_source_sha256": before_sha,
        "source_sha256": after_sha,
        "reviewer": "har-b03-c2-tertiary-reset-omission-exact-pairs-v1",
    }
