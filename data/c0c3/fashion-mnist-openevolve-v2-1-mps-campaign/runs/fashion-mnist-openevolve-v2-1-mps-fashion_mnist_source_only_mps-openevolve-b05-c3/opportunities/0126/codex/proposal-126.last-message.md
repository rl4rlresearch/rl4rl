MECHANISM: Verified decision-preserving ensemble calibration

HYPOTHESIS: Increasing only the outer sharpening to 1.10× will preserve all 9,206 predictions while reducing validation cross-entropy to approximately 0.224459.

INTENDED_EDIT: Multiply the averaged TTA log-probabilities by 1.10 instead of 1.05.

EVIDENCE: Reference Design 3 verified this exact configuration at 9,206 correct and 0.224459 cross-entropy, improving upon the current design’s tied correctness and 0.227083 cross-entropy.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE