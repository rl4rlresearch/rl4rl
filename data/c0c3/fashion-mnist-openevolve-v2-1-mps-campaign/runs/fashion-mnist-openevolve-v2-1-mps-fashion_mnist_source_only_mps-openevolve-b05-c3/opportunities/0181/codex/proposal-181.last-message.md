MECHANISM: Bracketed post-ensemble temperature calibration

HYPOTHESIS: A 1.178× final multiplier will preserve all 9,210 predictions while reducing validation cross-entropy below 0.22289956.

INTENDED_EDIT: Increase only the argmax-invariant final ensemble multiplier from 1.175× to 1.178×.

EVIDENCE: Cross-entropy improved from 0.22300627 at 1.15× to 0.22289956 at 1.175×, then worsened to 0.22291059 at 1.1875×; interpolation places the calibration minimum near 1.178×.

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.175)
=======
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
>>>>>>> REPLACE