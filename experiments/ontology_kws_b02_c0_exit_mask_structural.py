"""Structural source review for the canonical KWS B02-C0 early-exit family.

The rule is deliberately tied to the immutable source outside ``exit_mask``.  It
recognizes any pair whose only source difference is that method, so queue
extensions of this same source family do not need a per-proposal ledger.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap

from experiments.ontology_review_recurrent import source_sha

_CANONICAL_OUTER_SHA256 = "94074f39657f6a9c79d31f42fcf330f35fa318a871a363c43be18c483e5429c2"
_PLACEHOLDER = "    # EXIT_MASK_PLACEHOLDER\n"


def _split_exit_mask(source: str) -> tuple[str, str] | None:
    lines = source.splitlines(keepends=True)
    start = next((index for index, line in enumerate(lines) if re.match(r"    def exit_mask\(", line)), None)
    if start is None:
        return None
    end = next((index for index in range(start + 1, len(lines)) if re.match(r"(?:    )?def \w+\(", lines[index])), len(lines))
    return "".join(lines[:start] + [_PLACEHOLDER] + lines[end:]), "".join(lines[start:end])


def _outer_sha(sources: dict[str, str]) -> str | None:
    train = sources.get("train.py")
    if train is None:
        return None
    split = _split_exit_mask(train)
    if split is None:
        return None
    outer_sources = dict(sources)
    outer_sources["train.py"] = split[0]
    return hashlib.sha256(json.dumps(outer_sources, sort_keys=True).encode()).hexdigest()


def _exit_mask_evidence(source: str) -> tuple[int, str] | None:
    split = _split_exit_mask(source)
    if split is None:
        return None
    _, body = split
    try:
        tree = ast.parse(textwrap.dedent(body))
    except SyntaxError:
        return None
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "exit_mask"), None)
    if function is None or not any(isinstance(node, ast.Return) for node in ast.walk(function)):
        return None
    required = ("certified", "heuristic_exit", "tail_indices", "certificate_slack")
    if not all(token in body for token in required):
        return None
    line = source[:source.index(body)].count("\n") + 1
    digest = hashlib.sha256(body.encode()).hexdigest()
    return line, digest


def review_pair(before_sources: dict[str, str], after_sources: dict[str, str]):
    if source_sha(before_sources) == source_sha(after_sources):
        return None
    if _outer_sha(before_sources) != _CANONICAL_OUTER_SHA256 or _outer_sha(after_sources) != _CANONICAL_OUTER_SHA256:
        return None
    before = _exit_mask_evidence(before_sources["train.py"])
    after = _exit_mask_evidence(after_sources["train.py"])
    if before is None or after is None or before[1] == after[1]:
        return None
    before_line, before_digest = before
    after_line, after_digest = after
    return {
        "classification": "changing",
        "fingerprint": {
            "input_units": "frozen 20-band acoustic feature frames",
            "input_transform": "learned affine LayerNorm; source-fixed low-band fusion, high-band fusion, and tanh high-band contrast",
            "embedding": "16 recurrent acoustic features plus a one-coordinate high-band contrast summary",
            "position": "causal streaming frame order",
            "mixing": "one standard unidirectional 16-input, 98-hidden GRU initialized by a source-fixed 18-input, 99-hidden baseline projection",
            "routing": "fixed recurrent path; exit_mask conditionally halts eligible examples using source-defined certificate and heuristic masks",
            "state": "GRU hidden state, running output sum, scalar frame count, and retained high-band contrast summary",
            "feedforward": "99-to-7 contrast classifier expanded by a fixed 7-to-8 Helmert-style contrast map",
            "parameter_construction": "constructor deterministically projects a saved-RNG baseline GRU and classifier into the 16-input/98-hidden model and builds a fixed contrast expansion buffer",
            "sharing": "streaming and sequence paths share normalization, band fusion, GRU, summary, contrast classifier, and exit-mask policy",
            "normalization": "learned affine LayerNorm over acoustic frames; mean divides by count clamped below at one",
            "connectivity": "fused bands and contrast feed the GRU; temporal mean concatenates with contrast before classification and exit evaluation",
            "aggregation": "causal running arithmetic mean of GRU outputs",
            "output": "fixed 8-class logits reconstructed from seven learned contrast logits",
            "symmetry": "the fixed contrast expansion removes the common-logit degree of freedom",
            "conditional_compute": "canonical source-structural B02-C0 certificate/heuristic early-exit policy; exact after exit_mask SHA-256 is %s" % after_digest,
            "activation": "GRU sigmoid/tanh gates, tanh high-band contrast, and affine contrast classifier",
            "stochasticity": "training applies source-fixed Gaussian acoustic noise; inference exit_mask is deterministic",
            "bottleneck": "16 recurrent input features and 98 hidden units, with seven contrast logits expanded to eight output logits",
            "iteration": "one causal GRU scan with a source-defined deterministic exit decision at each eligible frame",
            "other": "Complete structural audit: every source outside KeywordGRU.exit_mask equals immutable canonical outer SHA-256 %s; before exit_mask SHA-256 is %s." % (_CANONICAL_OUTER_SHA256, before_digest),
            "acoustic_representation": "protected log-mel acoustic frames with source-fixed low/high-band pooling",
            "state_update": "standard GRU update followed by output-sum/count accumulation",
            "state_structure": "hidden [batch,1,98], running output sum [batch,98], count [batch,1], and high-band contrast [batch,1]",
            "temporal_schedule": "fixed source schedule: process the source-defined cropped acoustic window",
            "readout_history": "temporal mean of GRU output concatenated with the retained high-band contrast",
            "exit_policy": "certificate plus heuristic tail-frame early exit; exact after policy is bound by exit_mask SHA-256 %s" % after_digest,
        },
        "fingerprint_complete": True,
        "residual_components": [],
        "changed_components": ["routing", "conditional_compute", "output"],
        "notes": "Complete canonical B02-C0 structural review: the source maps are identical except KeywordGRU.exit_mask, whose deterministic certificate/heuristic early-exit policy changed.",
        "evidence": [{"file": "train.py", "line": after_line, "kind": "canonical outer-source hash plus parsed exit_mask source proof", "code": "KeywordGRU.exit_mask changed from SHA-256 %s to %s while all other source text remains bound to canonical outer SHA-256." % (before_digest, after_digest)}],
        "settings": {"canonical_outer_source_sha256": _CANONICAL_OUTER_SHA256, "before_exit_mask_sha256": before_digest, "after_exit_mask_sha256": after_digest},
        "family_label": "KWS B02-C0 / canonical contrast-GRU certificate and heuristic early-exit policy",
        "family_signature": {"state": "16-input 98-hidden GRU with causal mean and high-band contrast", "readout": "seven contrast logits expanded to eight classes", "exit": "parsed certificate/heuristic exit_mask under canonical immutable outer source"},
        "before_source_sha256": source_sha(before_sources),
        "source_sha256": source_sha(after_sources),
        "reviewer": "kws-b02-c0-canonical-exit-mask-structural-v1",
    }




