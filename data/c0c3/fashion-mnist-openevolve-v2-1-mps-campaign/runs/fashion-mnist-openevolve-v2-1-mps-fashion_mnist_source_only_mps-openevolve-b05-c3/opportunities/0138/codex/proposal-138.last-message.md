MECHANISM: Verified decision-preserving ensemble calibration

HYPOTHESIS: Increasing only the post-ensemble sharpening from 1.05× to 1.10× will preserve all 9,206 correct predictions while reducing validation cross-entropy toward 0.224459.

INTENDED_EDIT: Change the final TTA log-probability multiplier to 1.10 without altering training or ensemble argmaxes.

EVIDENCE: Reference Design 3 differs from the current design only in this multiplier and verified 9,206 correct with 0.224459 cross-entropy, improving the current tied-correctness cross-entropy of 0.227083.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE