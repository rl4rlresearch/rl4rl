MECHANISM: Incrementally relaxed confidence-gated final-frame skipping

HYPOTHESIS: Lowering the penultimate-step exit threshold from 0.85 to 0.80 will retain at least 85% validation accuracy while reducing recurrent MACs and mean recurrent steps.

INTENDED_EDIT: Exit after 24 of 25 scheduled steps when provisional classifier confidence is at least 0.80 instead of 0.85.

EVIDENCE: Successive threshold reductions from 0.97 through 0.85 preserved at least 85.03% accuracy while monotonically reducing total inference MACs; the latest 0.85 result achieved 85.15% accuracy and 682,366,944 MACs, motivating another incremental relaxation.

<<<<<<< SEARCH
        return confidence >= 0.85
=======
        return confidence >= 0.80
>>>>>>> REPLACE