MECHANISM: Quadratically refined evaluation-temperature calibration

HYPOTHESIS: Temperature 0.800713 will preserve all 9,328 argmax predictions while reducing validation cross-entropy below 0.191472039.

INTENDED_EDIT: Restore the best-count constant-smoothing design and sharpen its confidence-adaptive fused evaluation logits using the locally estimated optimal temperature.

EVIDENCE: Temperatures 0.8000, 0.8007, and 0.8050 produced cross-entropies 0.191472182, 0.191472039, and 0.191477223 with identical correct counts; quadratic interpolation places the local minimum near 0.800713.

<<<<<<< SEARCH
        return fused_log_probabilities / 0.90
=======
        return fused_log_probabilities / 0.800713
>>>>>>> REPLACE