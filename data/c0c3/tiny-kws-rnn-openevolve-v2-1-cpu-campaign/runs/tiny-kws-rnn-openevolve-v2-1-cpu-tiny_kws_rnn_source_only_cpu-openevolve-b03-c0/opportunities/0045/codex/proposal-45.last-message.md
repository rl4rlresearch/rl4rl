MECHANISM: Further relaxed confidence-gated final-frame skipping

HYPOTHESIS: Lowering the penultimate-step exit threshold from 0.90 to 0.85 will retain at least 85% validation accuracy while reducing recurrent MACs and mean recurrent steps.

INTENDED_EDIT: Exit after 24 of 25 scheduled steps when provisional classifier confidence is at least 0.85 instead of 0.90.

EVIDENCE: Lowering the threshold from 0.95 to 0.90 preserved 85.15% accuracy while reducing mean recurrent steps from 24.720 to 24.607 and total inference MACs from 688,035,432 to 684,836,040, supporting another incremental relaxation.

<<<<<<< SEARCH
        return confidence >= 0.90
=======
        return confidence >= 0.85
>>>>>>> REPLACE