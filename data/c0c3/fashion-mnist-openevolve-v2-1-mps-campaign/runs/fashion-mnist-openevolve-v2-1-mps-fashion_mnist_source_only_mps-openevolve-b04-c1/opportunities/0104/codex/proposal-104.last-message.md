MECHANISM: Delayed quadratic label-smoothing decay

HYPOTHESIS: Retaining more label smoothing during the second half while still reaching zero at training end will exceed 9,328 correct predictions by preserving useful late regularization without permanently biasing the final classifier toward soft targets.

INTENDED_EDIT: Replace the linear label-smoothing decay with a delayed quadratic decay; keep the dropout schedule and all other training behavior unchanged.

EVIDENCE: Ending label smoothing at 80% progress reduced validation correct from 9,328 to 9,316, while the linear full-duration decay remains best, motivating a runtime-neutral test in the opposite direction: slower soft-target removal.

<<<<<<< SEARCH
    label_smoothing = 0.02 * (1.0 - dropout_decay)
=======
    label_smoothing = 0.02 * (1.0 - dropout_decay * dropout_decay)
>>>>>>> REPLACE