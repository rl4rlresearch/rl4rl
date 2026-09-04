MECHANISM: Curriculum-coupled hard ensemble supervision

HYPOTHESIS: Using hard labels for the increasingly dominant ensemble loss while retaining 0.02 smoothing for individual views will exceed 9,330 correct predictions without adding computation.

INTENDED_EDIT: Remove label smoothing only from the ensemble loss, so the existing curriculum progressively shifts training toward the hard-label validation objective.

EVIDENCE: The best verified design reached 9,330 correct with ensemble weight rising from 0.5 to 1.0. Annealed smoothing targeted the same late hard-label alignment but timed out without contrary validation evidence; this change realizes that idea through the existing loss mixture.

<<<<<<< SEARCH
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=label_smoothing,
    )
=======
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.0,
    )
>>>>>>> REPLACE