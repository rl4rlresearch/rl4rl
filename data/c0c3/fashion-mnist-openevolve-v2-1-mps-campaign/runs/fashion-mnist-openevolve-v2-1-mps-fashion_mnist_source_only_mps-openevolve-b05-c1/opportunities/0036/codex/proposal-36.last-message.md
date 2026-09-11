MECHANISM: Fine-grained post-fusion temperature calibration

HYPOTHESIS: Scaling fused log-probabilities by 1.1063 will preserve all 9,249 predictions while reducing validation cross-entropy below 0.2150005993.

INTENDED_EDIT: Refine the inference-only probability-ensemble sharpening factor from 1.106 to 1.1063.

EVIDENCE: Scales 1.0, 1.1, and 1.15 yielded cross-entropies 0.216368, 0.215002, and 0.215228 with identical correctness; quadratic interpolation places the minimum near 1.10633, slightly above the current best scale.

<<<<<<< SEARCH
        return 1.106 * probabilities.clamp_min(1e-8).log()
=======
        return 1.1063 * probabilities.clamp_min(1e-8).log()
>>>>>>> REPLACE