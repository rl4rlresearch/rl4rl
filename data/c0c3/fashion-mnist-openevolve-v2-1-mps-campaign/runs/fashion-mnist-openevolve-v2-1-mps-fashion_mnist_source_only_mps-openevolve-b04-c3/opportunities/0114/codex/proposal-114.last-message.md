MECHANISM: Mild label-smoothed classification objective

HYPOTHESIS: Training with 0.05 label smoothing will increase validation_correct above 9,243 by reducing overconfident fitting within the fixed 100,000-example budget.

INTENDED_EDIT: Add mild label smoothing to the existing cross-entropy loss while preserving the verified architecture, augmentation, schedule, and inference ensemble.

EVIDENCE: Temperature and power-mean refinements repeatedly plateaued at exactly 9,243 correct, so an inexpensive training-objective change is the most direct next test capable of changing class rankings without the runtime overhead that accompanied augmentation and architectural experiments.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
>>>>>>> REPLACE