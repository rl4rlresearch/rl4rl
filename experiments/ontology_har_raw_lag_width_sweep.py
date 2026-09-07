"""Exact-source review for the retained HAR RawLagInteractionNet width sweep.

This is an enumeration of eleven recorded proposal pairs, not a rule for
``hiddenWidth`` edits.  Every entry is accepted only when both complete source
bundles match their recorded SHA-256 values.  Proposal 160, which switches the
selected model to a self-attention classifier, is deliberately absent.
"""
from __future__ import annotations

from experiments.ontology_review_recurrent import source_sha


# proposal, parent source SHA-256, candidate source SHA-256, parent width,
# candidate width.  These are proposals 158, 159, and 161 through 169 in the
# B01-C1 retained lineage.  Each full source diff changes only build_model's
# RawLagInteractionNet(hiddenWidth=...) argument.
EXACT_PAIRS = (
    (158, "44d24dddc0c77a9e4aeb25d4031b827bd358fe020bb36ffe22e055e5b2625841", "f2635aaf60995fb3e7de71c6a93826a24aeac33ebb18c448b0b3c0d9e5dc23b0", 21, 20),
    (159, "f2635aaf60995fb3e7de71c6a93826a24aeac33ebb18c448b0b3c0d9e5dc23b0", "e63060ad7980731e5fd0d99b2036b3dbfdc897e1a7406de106ffc43fa1e1577c", 20, 19),
    (161, "e63060ad7980731e5fd0d99b2036b3dbfdc897e1a7406de106ffc43fa1e1577c", "7c24b718ebd5e63c2e144386fa3c11a17271cc04649fd28000e23b6de6d543e4", 19, 18),
    (162, "7c24b718ebd5e63c2e144386fa3c11a17271cc04649fd28000e23b6de6d543e4", "51da108e9fb238de9c1d416487be24fe54eef22531992e09b077a0381199cef2", 18, 17),
    (163, "51da108e9fb238de9c1d416487be24fe54eef22531992e09b077a0381199cef2", "28c119f311a8e905b64c3cda5f1129f3975eca6c3936f4fab41f8c4347e63616", 17, 16),
    (164, "28c119f311a8e905b64c3cda5f1129f3975eca6c3936f4fab41f8c4347e63616", "b380e9218420145291f0c309c99e99ac59b7942a534d23a70b35b8ec59ac3b2a", 16, 15),
    (165, "b380e9218420145291f0c309c99e99ac59b7942a534d23a70b35b8ec59ac3b2a", "3c222cba434f73953b1b5a0f06c0814673a5ab2a1730cf49d12b32907ff8a919", 15, 14),
    (166, "3c222cba434f73953b1b5a0f06c0814673a5ab2a1730cf49d12b32907ff8a919", "6a55fbefbd62c429d873926785a71e25f8382888d73d276d94a02e5e4cde9cff", 14, 13),
    (167, "6a55fbefbd62c429d873926785a71e25f8382888d73d276d94a02e5e4cde9cff", "07f96890b93f3e0156d5139153ed6eff7bff2b3fc239df42e323099f843defce", 13, 12),
    (168, "07f96890b93f3e0156d5139153ed6eff7bff2b3fc239df42e323099f843defce", "f18b7ee32f70b6e883d9745c960ae8549c916b4274af2317b8c9c864201a2fa9", 12, 11),
    (169, "f18b7ee32f70b6e883d9745c960ae8549c916b4274af2317b8c9c864201a2fa9", "7de93846b8f04054ddd135e5ae45f82d8bef8b9f775e5de88462cded0f032b87", 11, 10),
)


def _fingerprint(hidden_width: int) -> dict[str, str]:
    """Complete ontology for any one of the explicitly bound pairs."""
    return {
        "input_units": "fixed 128-step, nine-channel tri-axial human-activity sensor windows",
        "input_transform": "the three ordered xyz sensor groups are augmented with their Euclidean magnitudes, producing 12 enriched channels; first differences form the motion branch",
        "embedding": "no learned temporal embedding; the classifier consumes deterministic whole-window raw, enriched-level, motion, and quarter-summary interactions",
        "position": "fixed temporal offsets, whole-window moments, and four equal temporal quarters; no positional embedding",
        "mixing": "fixed pairwise products of standardized enriched channels at zero and positive lags; no learned cross-channel mixer",
        "routing": "the selected build_model path always invokes RawLagInteractionNet; fixed loops enumerate level lags (1, 4, 8, 32), motion lags (1, 4, 8, 16, 32), and magnitude cadence lags (8, 32)",
        "state": "no recurrent, cached, or persistent deployment state",
        "feedforward": f"BatchNorm over 951 deterministic coordinates, then {hidden_width}-wide affine/GELU/dropout/affine classification head",
        "parameter_construction": "ordinary free affine classifier parameters only; upper-triangle index buffers are fixed and nonpersistent",
        "sharing": "each fixed lag uses the same standardized enriched or motion trajectories; the shared classifier applies to every example",
        "normalization": "per-window standardization for level and motion lag products, variance-stabilized magnitude cadence, and BatchNorm on concatenated features",
        "connectivity": "raw and magnitude statistics, enriched level correlations, enriched-difference correlations, and quarter summaries concatenate once before the BatchNorm/classifier head",
        "aggregation": "66 upper-triangular off-diagonal zero-lag enriched coordinates; four complete 12-by-12 directed level-lag tables; 66 zero-lag motion coordinates; five 12-wide motion-lag vectors; 120 independent quarter coordinates; and 63 raw/magnitude coordinates",
        "output": "six activity logits",
        "symmetry": "only zero-lag enriched and motion matrices use one upper-triangular off-diagonal copy; all positive level lags remain directed",
        "conditional_compute": "none: all lag and quarter loops have source-fixed bounds",
        "activation": "GELU in the classifier; square roots for magnitudes, standard deviations, and RMS; absolute value for motion",
        "stochasticity": "dropout with probability 0.1 in the classifier during training; no inference-time random branch",
        "bottleneck": f"the sole learned bottleneck is the {hidden_width}-wide classifier hidden layer after the fixed 951-coordinate representation",
        "iteration": "finite source-fixed lag loops and four temporal quarters; no recurrent scan",
        "other": "Complete exact-pair proof: the two source SHA-256 values bind the entire train.py diff. The only changed runtime source component is RawLagInteractionNet's selected hiddenWidth keyword in build_model.",
        "sensor_representation": "raw xyz, group magnitudes, enriched first differences, raw/magnitude moments, standardized level and motion lag interactions",
        "sensor_fusion": "three ordered xyz groups are fused only by their within-group magnitude calculation and fixed pairwise enriched-channel products",
        "temporal_operator": "whole-window moments plus fixed lag products at level lags 1/4/8/32, motion lags 1/4/8/16/32, magnitude cadence lags 8/32, and four equal quarters",
        "directionality": "positive lag level products preserve leading/trailing channel order; no recurrent direction",
        "state_update": "none",
        "state_structure": "none",
        "frequency_representation": "no spectral transform; temporal frequency is represented only by fixed lag products",
        "temporal_schedule": "128 samples split into four 32-sample quarters; all lag tuples are source-fixed",
        "readout_history": "whole-window statistics and fixed-offset pair products; no endpoint-only readout",
        "temporal_readout": "concatenated raw/magnitude statistics, level and motion lag features, and independent quarter summaries",
        "exit_policy": "fixed full-window classification; no early-exit hook",
    }


def review_pair(before_sources: dict[str, str], after_sources: dict[str, str]):
    """Recognize exactly one recorded RawLag width transition, if present."""
    before_sha = source_sha(before_sources)
    after_sha = source_sha(after_sources)
    match = next(
        (entry for entry in EXACT_PAIRS if entry[1:3] == (before_sha, after_sha)),
        None,
    )
    if match is None:
        return None

    proposal, _, _, before_width, after_width = match
    return {
        "classification": "preserving",
        "fingerprint": _fingerprint(after_width),
        "fingerprint_complete": True,
        "residual_components": [],
        "changed_components": [],
        "notes": (
            f"Exact source-pair proof for proposal {proposal}. The complete "
            "train.py diff changes only build_model's selected "
            f"RawLagInteractionNet hiddenWidth from {before_width} to "
            f"{after_width}; all representation, fixed-lag routing, "
            "normalization, and training-source operations are hash-bound and "
            "unchanged."
        ),
        "evidence": [{
            "file": "train.py",
            "line": 2884,
            "code": (
                "build_model returns RawLagInteractionNet(..., "
                f"hiddenWidth={after_width}, dropout=0.1); the bound parent "
                f"uses hiddenWidth={before_width}."
            ),
            "kind": "complete hash-bound RawLagInteractionNet width-sweep source-pair proof",
        }],
        "settings": {
            "proposal": proposal,
            "selected_model": "RawLagInteractionNet",
            "classifier_input_coordinates": 951,
            "hidden_width": {"before": before_width, "after": after_width},
            "dropout": 0.1,
            "level_lags": [1, 4, 8, 32],
            "motion_lags": [1, 4, 8, 16, 32],
            "magnitude_cadence_lags": [8, 32],
            "changed_source_components": [
                "build_model RawLagInteractionNet hiddenWidth keyword",
            ],
            "unchanged_ontology_components": [
                "input representation", "feature routing", "lag tuples",
                "normalization", "aggregation", "classifier topology",
                "training procedure",
            ],
        },
        "family_label": "Raw sensor lag-interaction statistics / compact nonlinear classifier",
        "family_signature": {
            "sensor_pipeline": "xyz grouped with magnitude enrichment and first differences",
            "state": "none",
            "interaction_features": "upper-triangular zero-lag and directed positive-lag enriched/motion products",
            "temporal_summary": "whole-window moments, fixed lag tuples, and four quarters",
            "head": "BatchNorm then affine/GELU/dropout/affine",
            "routing": "fixed source-lag loops",
        },
        "before_source_sha256": before_sha,
        "source_sha256": after_sha,
        "reviewer": "har-raw-lag-hidden-width-sweep-exact-pairs-v1",
    }
