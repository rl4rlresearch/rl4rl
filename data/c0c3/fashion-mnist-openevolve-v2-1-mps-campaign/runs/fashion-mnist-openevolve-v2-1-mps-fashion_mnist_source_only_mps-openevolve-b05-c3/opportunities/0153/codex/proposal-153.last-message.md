MECHANISM: Post-ensemble temperature sharpening

HYPOTHESIS: Increasing final sharpening from 1.10× to 1.15× will preserve the current 9,208 predictions while reducing validation cross-entropy below 0.224548.

INTENDED_EDIT: Increase only the argmax-invariant multiplier applied after the qualified half-weight centered-view ensemble.

EVIDENCE: Raising final sharpening from 1.05× to 1.10× preserved 9,206 correct while reducing cross-entropy from 0.227083 to 0.224459; the prior 1.15× attempt timed out and supplied no contrary performance evidence.

<<<<<<< SEARCH
        return (probability_sum / 9.0).clamp_min(1e-8).log().mul_(1.10)
=======
        return (probability_sum / 9.0).clamp_min(1e-8).log().mul_(1.15)
>>>>>>> REPLACE