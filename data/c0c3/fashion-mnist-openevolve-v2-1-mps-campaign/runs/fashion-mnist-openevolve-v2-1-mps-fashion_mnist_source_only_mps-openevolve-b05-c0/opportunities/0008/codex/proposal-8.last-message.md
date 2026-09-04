MECHANISM: Stronger target-distribution regularization

HYPOTHESIS: Increasing label smoothing from 0.05 to 0.10 will exceed 9,170 correct predictions by improving class-boundary generalization while preserving the successful architecture, schedule, and tail EMA.

INTENDED_EDIT: Change only the cross-entropy label-smoothing strength from 0.05 to 0.10.

EVIDENCE: Removing label smoothing reduced validation accuracy from 9,170 to 9,135 correct despite lowering cross-entropy, indicating that regularization—not stronger finite-budget fitting—is more important for the primary accuracy objective.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.10)
>>>>>>> REPLACE