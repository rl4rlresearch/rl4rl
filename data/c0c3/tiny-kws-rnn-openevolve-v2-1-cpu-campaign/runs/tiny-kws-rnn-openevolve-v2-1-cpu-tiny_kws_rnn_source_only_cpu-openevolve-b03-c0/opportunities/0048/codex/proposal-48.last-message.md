MECHANISM: Incrementally relaxed confidence-gated final-frame skipping

HYPOTHESIS: Lowering the penultimate-step exit threshold from 0.75 to 0.70 will retain at least 85% validation accuracy while reducing recurrent MACs and mean recurrent steps.

INTENDED_EDIT: Exit after 24 of 25 scheduled steps when provisional classifier confidence is at least 0.70 instead of 0.75.

EVIDENCE: Threshold reductions from 0.97 through 0.75 preserved at least 85.03% accuracy and monotonically reduced inference MACs; the 0.75 design still achieved 85.15% accuracy while lowering mean recurrent steps to 24.396.

<<<<<<< SEARCH
        return confidence >= 0.75
=======
        return confidence >= 0.70
>>>>>>> REPLACE