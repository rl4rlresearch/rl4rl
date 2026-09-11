MECHANISM: Local quadratic refinement of agreement-conditioned calibration

HYPOTHESIS: A 7.727% disagreement adjustment will preserve all 9,287 predictions through positive per-image scaling while reducing validation cross-entropy below 0.20668936004638672.

INTENDED_EDIT: Replace the current 15% agreement-conditioning strength with the quadratically estimated 7.727% optimum, leaving training and pooled logits unchanged.

EVIDENCE: The verified 0%, 7.708%, and 15% adjustments yielded cross-entropies of 0.20670592765808105, 0.20668936004638672, and 0.20670405883789061; their quadratic minimum is near 7.727%, and its only prior verification timed out.

<<<<<<< SEARCH
        calibration = 1.22775 * (0.85 + 0.15 * view_agreement)
=======
        calibration = 1.22775 * (0.92273 + 0.07727 * view_agreement)
>>>>>>> REPLACE