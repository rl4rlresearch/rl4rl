MECHANISM: Stronger decision-preserving ensemble calibration

HYPOTHESIS: Increasing the outer sharpening factor from 1.10× to 1.15× will preserve all 9,206 argmax predictions while reducing validation cross-entropy below 0.224459.

INTENDED_EDIT: Increase only the post-ensemble log-probability multiplier; training and per-view TTA remain unchanged.

EVIDENCE: Raising outer sharpening from 1.05× to 1.10× preserved 9,206 correct predictions and reduced cross-entropy from 0.227083 to 0.224459; the prior 1.15× attempt timed out and supplied no contrary performance evidence.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.15)
>>>>>>> REPLACE