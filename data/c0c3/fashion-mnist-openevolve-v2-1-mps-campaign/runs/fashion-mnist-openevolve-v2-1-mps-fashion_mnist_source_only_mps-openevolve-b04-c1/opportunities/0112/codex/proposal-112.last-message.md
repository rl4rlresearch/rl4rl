MECHANISM: Late hard-label dropout floor

HYPOTHESIS: Retaining 0.05 classifier dropout at training end will exceed 9,328 correct predictions by regularizing late hard-label optimization without the soft-target bias that made slower label-smoothing decay underperform.

INTENDED_EDIT: Keep the existing early dropout and label-smoothing schedules, but anneal classifier dropout from 0.15 to 0.05 instead of zero during the second half.

EVIDENCE: Ending label smoothing earlier reduced correct predictions to 9,316, indicating late regularization matters, while retaining more smoothing reached only 9,325; dense-head widening also fell to 9,300, motivating unbiased late classifier regularization rather than more capacity or softer targets.

<<<<<<< SEARCH
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
=======
    model.classifier[3].p = (
        0.15 * (1.0 - dropout_decay)
        + 0.05 * dropout_decay
    )
>>>>>>> REPLACE