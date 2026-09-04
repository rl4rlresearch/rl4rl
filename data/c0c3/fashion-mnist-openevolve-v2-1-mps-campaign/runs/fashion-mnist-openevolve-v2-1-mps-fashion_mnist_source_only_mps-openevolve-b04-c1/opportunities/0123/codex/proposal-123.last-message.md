MECHANISM: Front-loaded label smoothing

HYPOTHESIS: Increasing label smoothing from 0.02 to 0.03 only before EMA collection will exceed 9,328 correct predictions by strengthening early representation regularization while preserving the successful late smoothing decay.

INTENDED_EDIT: Use 0.03 label smoothing during the first half of training, then retain the existing 0.02-to-zero schedule.

EVIDENCE: Ending smoothing earlier fell to 9,316 correct, showing early regularization is valuable, while retaining more smoothing late reached only 9,325; this isolates additional smoothing to the pre-EMA phase.

<<<<<<< SEARCH
    label_smoothing = 0.02 * (1.0 - dropout_decay)
=======
    if progress < 0.50:
        label_smoothing = 0.03
    else:
        label_smoothing = 0.02 * (1.0 - dropout_decay)
>>>>>>> REPLACE