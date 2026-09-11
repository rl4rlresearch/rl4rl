MECHANISM: Convex post-ensemble temperature refinement

HYPOTHESIS: A 1.17792× final multiplier will preserve all 9,210 argmax predictions while reducing validation cross-entropy below 0.2228984337.

INTENDED_EDIT: Replace the current 1.1875× final calibration with the estimated minimum of the bracketed cross-entropy curve.

EVIDENCE: Cross-entropy was 0.2228995598 at 1.175×, improved to 0.2228984337 at 1.178×, and worsened to 0.2229105946 at 1.1875×; local quadratic interpolation places the minimum near 1.17792×, and positive scaling cannot change argmax predictions.

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.1875)
=======
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.17792)
>>>>>>> REPLACE