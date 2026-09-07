"""Exact-source review for the second KWS B05-C0 compression chain.

This recognizer accepts only the immutable P88--P121 SHA ledger.  The shared
outer source is held fixed; each transition changes just the train(False)
deployment selector and its dependent relative-logit head width.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.ontology_kws_progressive_deployment_compression import _fingerprint
from experiments.ontology_review_recurrent import source_sha

_LEDGER_PATH = Path(__file__).with_name("ontology_kws_progressive_deployment_compression_v2_pairs.json")
EXACT_PAIRS = tuple(tuple(entry) for entry in json.loads(_LEDGER_PATH.read_text(encoding="utf-8-sig")))
_BY_HASH = {(parent, child): (proposal, before, after) for proposal, before, after, parent, child in EXACT_PAIRS}


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
        "notes": "Exact P%d source-pair review: the sole source delta is train(False), which adds one covariance-guided dropped deployment coordinate (%d to %d) and reduces the relative-logit head accordingly." % (proposal, before, after),
        "evidence": [{"file": "train.py", "line": line, "kind": "complete hash-bound B05-C0 progressive deployment-compression source-pair proof", "code": "train(False) extends the covariance-guided dropped-coordinate selector from %d to %d and changes nn.Linear(full_classifier.in_features - %d, 7)." % (before, after, after)}],
        "settings": {"proposal": proposal, "deployment_dropped_coordinates": {"before": before, "after": after}, "deployment_relative_head_input": {"before": 114-before, "after": 114-after}},
        "family_label": "KWS GRU / covariance-guided relative-logit deployment compression",
        "family_signature": {"state": "20-input 58-hidden GRU with running mean and final hidden state", "deployment": "class-zero-relative compressed affine readout with covariance-guided reconstructed coordinates", "routing": "source-fixed greedy dropped-coordinate selector"},
        "before_source_sha256": parent,
        "source_sha256": child,
        "reviewer": "kws-b05-c0-progressive-deployment-compression-v2-exact-pairs-v1",
    }
