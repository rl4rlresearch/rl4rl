MECHANISM: Accelerated late label-smoothing removal

HYPOTHESIS: Quadratically annealing label smoothing after mid-training will exceed 9,328 correct predictions by emphasizing hard-label separation earlier during EMA collection while retaining the successful 0.02 early regularization.

INTENDED_EDIT: Preserve the existing dropout schedule but change label smoothing from a linear to a quadratic decay over the second half of training.

EVIDENCE: Annealing late label smoothing to zero produced the strongest 9,328-correct design, whereas stronger persistent weight decay reduced accuracy; this motivates removing soft-target bias faster without adding computation or persistent regularization.

<<<<<<< SEARCH
    label_smoothing = 0.02 * (1.0 - dropout_decay)
=======
    label_smoothing = 0.02 * (1.0 - dropout_decay) ** 2
>>>>>>> REPLACE