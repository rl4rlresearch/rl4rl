MECHANISM: Incrementally relaxed confidence-gated final-frame skipping

HYPOTHESIS: Lowering the penultimate-step exit threshold from 0.80 to 0.75 will retain at least 85% validation accuracy while reducing recurrent MACs and mean recurrent steps.

INTENDED_EDIT: Exit after 24 of 25 scheduled steps when provisional classifier confidence is at least 0.75 instead of 0.80.

EVIDENCE: Successive threshold reductions from 0.97 through 0.80 preserved at least 85.03% accuracy while monotonically reducing inference MACs; the 0.80 design still achieved 85.15% accuracy, supporting one further incremental relaxation.

<<<<<<< SEARCH
        return confidence >= 0.80
=======
        return confidence >= 0.75
>>>>>>> REPLACE