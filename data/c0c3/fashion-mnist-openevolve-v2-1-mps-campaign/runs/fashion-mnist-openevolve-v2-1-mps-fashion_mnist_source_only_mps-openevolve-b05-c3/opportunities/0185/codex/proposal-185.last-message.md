MECHANISM: Convex post-ensemble temperature refinement

HYPOTHESIS: A 1.17792× final multiplier will preserve all 9,210 argmax predictions while reducing validation cross-entropy below 0.2228984337.

INTENDED_EDIT: Refine only the argmax-invariant final ensemble calibration from 1.178× to the estimated local minimum, leaving training and TTA unchanged.

EVIDENCE: Cross-entropy improved from 0.2228995598 at 1.175× to 0.2228984337 at 1.178×, then worsened to 0.2229105946 at 1.1875×; quadratic interpolation places the minimum near 1.17792×, and its previous verification timed out without contrary accuracy evidence.

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
=======
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.17792)
>>>>>>> REPLACE