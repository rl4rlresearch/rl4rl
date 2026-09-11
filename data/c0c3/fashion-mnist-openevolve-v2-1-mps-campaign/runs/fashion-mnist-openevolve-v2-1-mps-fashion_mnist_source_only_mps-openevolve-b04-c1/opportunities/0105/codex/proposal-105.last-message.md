MECHANISM: Delayed quadratic label-smoothing decay

HYPOTHESIS: Retaining more label smoothing during late EMA collection while still reaching zero at training end will exceed 9,328 correct predictions.

INTENDED_EDIT: Replace linear label-smoothing decay with quadratic decay while leaving dropout and all other training behavior unchanged.

EVIDENCE: Ending smoothing at 80% progress reduced validation correct from 9,328 to 9,316, favoring later regularization; the prior quadratic-decay verification timed out and supplied no contrary accuracy evidence.

<<<<<<< SEARCH
    label_smoothing = 0.02 * (1.0 - dropout_decay)
=======
    label_smoothing = 0.02 * (1.0 - dropout_decay * dropout_decay)
>>>>>>> REPLACE