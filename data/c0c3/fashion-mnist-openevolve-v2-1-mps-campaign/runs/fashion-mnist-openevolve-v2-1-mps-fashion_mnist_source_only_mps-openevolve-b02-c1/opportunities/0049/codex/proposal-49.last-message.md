MECHANISM: Post-ensemble temperature calibration

HYPOTHESIS: Softening the evaluation ensemble with temperature 1.05 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.

INTENDED_EDIT: Divide the final evaluation logits by 1.05 without changing training, parameters, augmentation, averaging, or class predictions.

EVIDENCE: The current implementation has the best observed correct count, while stronger tail interpolation slightly reduced cross-entropy but damaged accuracy; temperature scaling targets only the tie-break metric and leaves every argmax unchanged.

<<<<<<< SEARCH
        return ensemble - math.log(12.0)
=======
        return (ensemble - math.log(12.0)) / 1.05
>>>>>>> REPLACE