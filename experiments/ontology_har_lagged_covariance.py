"""Exact-source review for HAR proposal 81's zero-lag covariance packing.

This is deliberately a pair declaration, not a reusable AST rewrite rule.  It
recognizes only the retained proposal-80 parent and proposal-81 candidate
source bundles.  The candidate packs a symmetric zero-lag covariance matrix
into its upper triangle; lagged matrices remain directed and fully flattened.
"""
from __future__ import annotations

from experiments.ontology_review_recurrent import source_sha


P80_PARENT_SHA256 = "2f1b748cf9d61ee6ec87e1b9e8c544b2906f2d017d4677ac469a83da5dbb2e73"
P81_CANDIDATE_SHA256 = "0049ae64c0d4237138f7d2d7d34ac24bf6d461ce4a58529afc501a5f9bafd074"


def _fingerprint() -> dict[str, str]:
    """Complete reviewed ontology for the hash-bound proposal-81 candidate."""
    return {
        "input_units": "fixed 128-step, nine-channel tri-axial human-activity sensor windows",
        "input_transform": "three xyz sensor groups are augmented with per-group magnitude and frame-to-frame xyz difference, producing 21 channels; raw magnitude signals are also centered for cadence correlations",
        "embedding": "a learned three-group temporal Conv1d stem maps the 21 enriched channels to 15 latent trajectories",
        "position": "translation-equivariant temporal convolutions with fixed kernel/stride/padding; no positional embedding",
        "mixing": "three-group stem communication, then dense learned 1x1 cross-sensor mixing, then depthwise temporal filtering; each update is residual GELU",
        "routing": "fixed all-path computation: every window forms upper-triangular zero-lag covariance, full directed latent cross-covariance at lags 2 and 8, and both raw magnitude-cadence lags",
        "state": "no recurrent or persistent deployment state",
        "feedforward": "LayerNorm over concatenated summaries, then 30-wide affine/GELU/dropout/affine classifier",
        "parameter_construction": "ordinary free convolution, normalization and classifier parameters; zero-lag triangle indices are a fixed nonpersistent buffer and do not construct learned parameters",
        "sharing": "the same group stem, dense mixer and depthwise filter serve all time positions; lag branches reuse the normalized latent trajectories",
        "normalization": "BatchNorm after stem, dense mixing and temporal filtering; latent trajectories are additionally standardized by their per-window mean and standard deviation before covariance",
        "connectivity": "enriched group channels feed grouped temporal convolution, dense cross-sensor residual mixing and depthwise temporal residual filtering; latent and raw-statistic branches concatenate only before the head",
        "aggregation": "latent temporal mean, standard deviation, maximum and four-bin means; the symmetric 15-by-15 upper-triangular zero-lag covariance contributes its 120 coordinates once, while complete directed 15-by-15 cross-covariance matrices remain at lags 2 and 8; raw mean/std/motion/peak/RMS, magnitude summaries, and magnitude autocorrelations at lags 8 and 32",
        "output": "six activity logits",
        "symmetry": "zero-lag latent covariance is symmetric and is represented once through its upper triangle; the input remains three ordered xyz groups, and lag-2/lag-8 matrices retain ordered leading/trailing latent-coordinate pairs",
        "conditional_compute": "none: fixed loops range only over source-fixed lag tuples",
        "activation": "GELU residual updates; square root for magnitudes/RMS/standard deviations; absolute value for motion; no recurrent gate",
        "stochasticity": "dropout in the classifier during training; no inference-time random branch",
        "bottleneck": "15 latent channels after the grouped stem; 120 unique zero-lag covariance coordinates, two full 15-by-15 directed lag matrices, seven 15-wide latent summary groups, and 63 raw-statistic coordinates",
        "iteration": "finite feedforward temporal convolutions and source-fixed lag loops over (0,2,8) and (8,32), with no recurrent scan",
        "other": "Complete exact-pair proof for proposal 81: the source hashes bind all edits to the retained parent/candidate pair; the only changed runtime observations are the fixed upper-triangle buffer and zero-lag extraction, with dependent feature-head dimensions.",
        "sensor_representation": "raw xyz sensors, per-group magnitudes, xyz deltas, latent lagged cross-covariances, and raw magnitude cadence correlations",
        "sensor_fusion": "three ordered xyz groups are fused first through group-wise magnitude/delta construction and a grouped stem, then through a dense learned 1x1 latent mixer",
        "temporal_operator": "grouped stride-four temporal convolution followed by depthwise kernel-five residual temporal filtering; lagged products use fixed offsets 0, 2, 8 for latents and 8, 32 for magnitude cadence",
        "directionality": "noncausal full-window temporal summaries with a symmetric zero-lag covariance and directed lagged products; no forward/backward recurrent direction",
        "state_update": "none",
        "state_structure": "none",
        "frequency_representation": "no spectral or frequency transform; the only fixed offsets are temporal lags",
        "temporal_schedule": "group stem kernel 7, stride 4, padding 3; depthwise filter kernel 5, padding 2; fixed summary bins reshape the 32 latent tokens into four groups of eight",
        "readout_history": "whole-window raw and latent summary statistics plus fixed-lag pairwise temporal products; no endpoint-only or recurrent history",
        "temporal_readout": "whole-window raw and latent mean/std/max, four-bin latent means, upper-triangular zero-lag covariance, complete directed cross-covariances at lags 2 and 8, and raw magnitude autocorrelations at two lags",
        "exit_policy": "fixed full-window classification; no early-exit hook",
    }


def review_pair(before_sources: dict[str, str], after_sources: dict[str, str]):
    """Recognize only the recorded proposal-80 -> proposal-81 transition."""
    before_sha = source_sha(before_sources)
    after_sha = source_sha(after_sources)
    if (before_sha, after_sha) != (P80_PARENT_SHA256, P81_CANDIDATE_SHA256):
        return None

    return {
        "classification": "preserving",
        "fingerprint": _fingerprint(),
        "fingerprint_complete": True,
        "residual_components": [],
        "changed_components": [],
        "notes": (
            "Exact source-pair proof for proposal 81. The zero-lag matrix is "
            "formed from normalized latent trajectories against itself, so entry "
            "(i, j) equals entry (j, i). The candidate retains each diagonal and "
            "one copy of every off-diagonal entry through torch.triu_indices; "
            "lags 2 and 8 remain full directed matrices. The complete source diff "
            "also changes only the dependent LayerNorm/classifier input width and "
            "the classifier's hidden width (28 to 30), which are feedforward "
            "coordinate/capacity settings under the retained ontology."
        ),
        "evidence": [{
            "file": "train.py",
            "line": 1481,
            "code": (
                "torch.triu_indices(width, width, offset=0) registers zeroLagRows "
                "and zeroLagColumns; only lag == 0 indexes correlation[:, rows, "
                "columns], while lags 2 and 8 retain correlation.flatten(1). The "
                "feature dimension changes from 843 to 738 and the affine hidden "
                "width from 28 to 30."
            ),
            "kind": "complete hash-bound proposal-80-to-proposal-81 source-pair proof",
        }],
        "settings": {
            "proposal": 81,
            "latent_width": 15,
            "latent_covariance_lags": [0, 2, 8],
            "zero_lag_coordinates": {"before": 225, "after": 120, "removed_duplicate_off_diagonals": 105},
            "directed_lag_coordinates": {"lag_2": 225, "lag_8": 225},
            "classifier_input": {"before": 843, "after": 738},
            "classifier_hidden_width": {"before": 28, "after": 30},
            "changed_source_components": [
                "fixed nonpersistent upper-triangle index buffers",
                "zero-lag covariance extraction",
                "dependent feature-normalization input width",
                "first classifier affine hidden width",
                "final classifier affine input width",
            ],
        },
        "family_label": "Grouped temporal CNN / dense cross-sensor mixer / symmetric zero-lag and directed lagged latent covariance",
        "family_signature": {
            "sensor_pipeline": "xyz+magnitude+delta group enrichment; grouped temporal stem; dense pointwise cross-sensor mixer; depthwise temporal refinement",
            "state": "none",
            "latent_summary": "mean/std/max/four-bin means plus upper-triangular zero-lag covariance and full directed lagged cross-covariances at 2,8",
            "raw_summary": "xyz/magnitude moments and motion plus magnitude cadence correlations at 8,32",
            "head": "LayerNorm then GELU affine classifier",
            "routing": "fixed source-lag loops",
        },
        "before_source_sha256": before_sha,
        "source_sha256": after_sha,
        "reviewer": "har-lagged-cross-covariance-zero-lag-exact-pair-v1",
    }
