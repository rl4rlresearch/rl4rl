MECHANISM: Modestly strengthened label smoothing

HYPOTHESIS: Increasing label smoothing from 0.015 to 0.020 will exceed 9,166 correct predictions by strengthening useful target regularization without introducing mixup’s input-distribution distortion.

INTENDED_EDIT: Use constant 0.020 label smoothing while preserving the architecture, augmentation, optimizer, schedule, EMA, and evaluation behavior.

EVIDENCE: Annealing smoothing toward zero reduced correctness from 9,166 to 9,160, and weaker dropout reached only 9,141, indicating that reduced regularization is harmful; the small increase avoids the substantially stronger soft-target intervention that made mixup fall to 9,130.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.015)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.020)
>>>>>>> REPLACE