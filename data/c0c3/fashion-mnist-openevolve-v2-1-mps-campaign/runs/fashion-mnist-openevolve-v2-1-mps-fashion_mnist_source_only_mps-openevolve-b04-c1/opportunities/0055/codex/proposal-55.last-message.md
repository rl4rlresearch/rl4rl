MECHANISM: EMA-aligned label-smoothing annealing

HYPOTHESIS: Holding label smoothing at 0.02 for the first half of training and annealing it to zero during EMA collection will exceed 9,323 correct predictions by removing late soft-target bias while retaining early regularization.

INTENDED_EDIT: Reuse the successful dropout-decay schedule to linearly reduce label smoothing from 0.02 to zero over the final half of training.

EVIDENCE: Annealing dropout during the final half improved validation correct from 9,316 to 9,323, indicating that removing regularization during late optimization and EMA collection is beneficial; this tests the same principle for the remaining loss regularizer without changing runtime or capacity.

<<<<<<< SEARCH
    logits = model(images)
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
=======
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    return F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE