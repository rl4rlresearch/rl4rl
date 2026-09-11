MECHANISM: Post-ensemble temperature sharpening

HYPOTHESIS: Increasing the final multiplier to 1.20× will preserve all 9,210 argmax predictions while reducing validation cross-entropy below 0.2230063.

INTENDED_EDIT: Increase only the argmax-invariant final ensemble multiplier from 1.15× to 1.20×.

EVIDENCE: Raising final sharpening from 1.10× to 1.15× previously retained 9,210 correct predictions and reduced cross-entropy; 1.20× remains unresolved only because its verifications timed out.

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.15)
=======
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.20)
>>>>>>> REPLACE