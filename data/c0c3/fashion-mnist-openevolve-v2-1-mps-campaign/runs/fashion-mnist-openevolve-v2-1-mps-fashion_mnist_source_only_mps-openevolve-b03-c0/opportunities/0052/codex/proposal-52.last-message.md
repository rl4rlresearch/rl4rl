MECHANISM: Reduced fixed label smoothing

HYPOTHESIS: Reducing label smoothing from 0.04 to 0.02 will exceed 9,319 correct predictions by preserving regularization while allowing stronger class margins during the limited two-pass exposure.

INTENDED_EDIT: Use fixed 0.02 label smoothing; retain the verified 0.80 evaluation temperature and all other training behavior.

EVIDENCE: Evaluation sharpening from 0.95 through 0.80 consistently lowered cross-entropy without changing 9,319 argmax predictions, indicating underconfident logits; the annealed-smoothing attempt timed out, so this isolates a lower smoothing strength without adding scheduling overhead.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
>>>>>>> REPLACE