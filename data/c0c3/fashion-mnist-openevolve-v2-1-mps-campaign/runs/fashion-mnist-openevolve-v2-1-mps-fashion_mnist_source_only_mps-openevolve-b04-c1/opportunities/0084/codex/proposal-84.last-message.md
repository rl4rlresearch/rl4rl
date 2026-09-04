MECHANISM: Persistent deterministic weight shrinkage

HYPOTHESIS: Increasing AdamW weight decay to 1e-2 will exceed 9,328 correct predictions by controlling late co-adaptation after dropout and label smoothing begin annealing away.

INTENDED_EDIT: Raise AdamW’s weight decay from 2e-4 to 1e-2 while preserving the winning architecture, learning-rate schedule, EMA, and stochastic-regularization schedules.

EVIDENCE: Expanding the classifier to 249,618 parameters reduced validation correct from 9,328 to 9,300, suggesting additional capacity is not beneficial; stronger persistent regularization tests that signal without adding runtime-heavy operations.

<<<<<<< SEARCH
        weight_decay=2e-4,
=======
        weight_decay=1e-2,
>>>>>>> REPLACE