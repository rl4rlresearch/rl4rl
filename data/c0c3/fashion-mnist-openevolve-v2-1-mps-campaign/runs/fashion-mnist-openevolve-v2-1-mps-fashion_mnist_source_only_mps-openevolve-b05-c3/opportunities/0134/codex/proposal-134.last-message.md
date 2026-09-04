MECHANISM: Verified decision-preserving ensemble calibration

HYPOTHESIS: Increasing only the post-ensemble sharpening from 1.05× to 1.10× will preserve 9,206 correct predictions while reducing validation cross-entropy from 0.227083 toward 0.224459.

INTENDED_EDIT: Change the final TTA ensemble log-probability multiplier to 1.10, leaving training and prediction argmaxes unchanged.

EVIDENCE: Reference Design 3 verified this exact configuration at 9,206 correct and 0.224459 cross-entropy, improving the current design’s tied-correctness cross-entropy of 0.227083.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE