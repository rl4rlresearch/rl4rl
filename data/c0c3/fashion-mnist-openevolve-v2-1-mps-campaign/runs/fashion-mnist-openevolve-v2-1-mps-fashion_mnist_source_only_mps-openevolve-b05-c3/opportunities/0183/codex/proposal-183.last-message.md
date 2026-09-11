MECHANISM: Verified post-ensemble temperature calibration

HYPOTHESIS: Applying the verified 1.178× final multiplier will preserve all 9,210 argmax predictions while reducing validation cross-entropy from 0.22300627 to approximately 0.22289843.

INTENDED_EDIT: Increase only the argmax-invariant final ensemble multiplier from 1.15× to the best verified 1.178× setting.

EVIDENCE: Reference Design 1 achieved the highest available validation_score, retaining 9,210 correct predictions while improving cross-entropy to 0.2228984337; nearby 1.175× and 1.1875× settings were both worse.

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.15)
=======
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
>>>>>>> REPLACE