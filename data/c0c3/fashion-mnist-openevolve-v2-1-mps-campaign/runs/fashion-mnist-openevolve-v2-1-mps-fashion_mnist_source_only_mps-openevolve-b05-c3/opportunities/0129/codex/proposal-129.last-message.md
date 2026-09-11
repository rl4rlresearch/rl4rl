MECHANISM: Conservative decision-preserving ensemble calibration

HYPOTHESIS: Increasing post-ensemble sharpening from 1.10× to 1.125× will preserve all 9,206 argmax predictions while reducing validation cross-entropy below 0.224459.

INTENDED_EDIT: Apply a midpoint calibration between the verified 1.10× factor and the repeatedly timed-out 1.15× proposal, leaving training and TTA aggregation unchanged.

EVIDENCE: Raising outer sharpening from 1.05× to 1.10× preserved 9,206 correct predictions and reduced cross-entropy from 0.227083 to 0.224459; the 1.15× attempts produced no performance evidence because verification timed out.

<<<<<<< SEARCH
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
=======
        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.125)
>>>>>>> REPLACE