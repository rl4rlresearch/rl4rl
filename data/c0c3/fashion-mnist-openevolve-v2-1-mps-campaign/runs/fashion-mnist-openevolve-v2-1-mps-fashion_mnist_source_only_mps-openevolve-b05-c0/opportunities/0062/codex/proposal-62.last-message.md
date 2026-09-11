MECHANISM: Low-strength label-smoothing refinement

HYPOTHESIS: Reducing label smoothing from 0.05 to 0.04 will exceed 9,258 correct predictions by slightly reducing target bias while retaining the established benefit of mild regularization.

INTENDED_EDIT: Change only the training cross-entropy label-smoothing coefficient from 0.05 to 0.04.

EVIDENCE: Fixed 0.05 smoothing outperformed both hard targets and 0.10 smoothing; the prior 0.04 attempt timed out, so it supplied no contradictory accuracy evidence and remains the closest runtime-neutral refinement.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
>>>>>>> REPLACE