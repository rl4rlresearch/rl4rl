MECHANISM: Stronger decision-preserving ensemble calibration

HYPOTHESIS: Increasing only the outer sharpening factor from 1.05× to 1.15× will preserve all 9,206 argmax predictions while reducing validation cross-entropy below 0.224459.

INTENDED_EDIT: Sharpen the averaged TTA log-probabilities by 1.15×; training and per-view sharpening remain unchanged.

EVIDENCE: Raising the outer factor from 1.05× to 1.10× preserved 9,206 correct predictions and reduced cross-entropy from 0.227083 to 0.224459. Prior 1.15× attempts timed out during training, which provides no contrary calibration evidence.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.15)
>>>>>>> REPLACE