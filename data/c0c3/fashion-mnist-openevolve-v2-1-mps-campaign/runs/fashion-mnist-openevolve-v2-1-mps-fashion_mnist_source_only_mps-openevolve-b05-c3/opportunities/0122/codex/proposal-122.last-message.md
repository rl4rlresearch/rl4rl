MECHANISM: Verified decision-preserving ensemble calibration

HYPOTHESIS: Increasing only the post-ensemble sharpening from 1.05× to 1.10× will preserve 9,206 correct predictions while reducing validation cross-entropy from 0.227083 to approximately 0.224459.

INTENDED_EDIT: Apply the strongest verified 1.10× outer calibration while retaining the current architecture, training procedure, EMA, and 1.05× per-view sharpening.

EVIDENCE: Reference Design 3 verified this exact configuration at 9,206 correct and 0.224459 cross-entropy, outperforming the current design’s tied correctness and 0.227083 cross-entropy.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE