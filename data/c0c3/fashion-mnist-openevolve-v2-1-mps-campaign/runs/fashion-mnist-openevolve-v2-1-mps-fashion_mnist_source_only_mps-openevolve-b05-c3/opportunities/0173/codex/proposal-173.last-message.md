MECHANISM: Post-ensemble temperature calibration

HYPOTHESIS: Increasing the final log-probability multiplier from 1.15× to 1.20× will preserve all 9,210 argmax predictions while reducing validation cross-entropy below 0.2230063.

INTENDED_EDIT: Sharpen only the final eight-view ensemble output, leaving training, per-view weighting, parameters, and inference workload unchanged.

EVIDENCE: Raising final sharpening from 1.10× to 1.15× preserved 9,210 correct predictions and reduced cross-entropy from 0.2253669 to 0.2237609; the prior 1.20× attempt only timed out, and positive final scaling is argmax-invariant.

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.15)
=======
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.20)
>>>>>>> REPLACE