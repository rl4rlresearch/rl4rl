MECHANISM: Verified post-ensemble temperature calibration

HYPOTHESIS: Replacing 1.1875× with the verified 1.178× final multiplier will preserve all 9,210 argmax predictions while reducing validation cross-entropy from 0.22291059 to approximately 0.22289843.

INTENDED_EDIT: Restore the highest-scoring verified final calibration without changing training, parameters, or inference workload.

EVIDENCE: Reference Design 1 achieved the best available validation_score with 9,210 correct and 0.2228984337 cross-entropy; both nearby 1.175× and the current 1.1875× multipliers produced higher cross-entropy.

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.1875)
=======
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
>>>>>>> REPLACE