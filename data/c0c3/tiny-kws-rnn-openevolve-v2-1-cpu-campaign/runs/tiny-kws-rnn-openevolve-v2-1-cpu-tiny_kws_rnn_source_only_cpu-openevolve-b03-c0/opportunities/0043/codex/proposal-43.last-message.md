MECHANISM: Moderately relaxed confidence-gated final-frame skipping

HYPOTHESIS: Lowering the penultimate-step exit threshold from 0.97 to 0.95 will preserve at least 85% validation accuracy while increasing the fraction of examples that skip the final recurrent step and reducing exact inference MACs.

INTENDED_EDIT: Exit after 24 of 25 scheduled steps when provisional classifier confidence is at least 0.95 instead of 0.97.

EVIDENCE: The 0.97 gate preserved the 25-step model’s exact 85.03% accuracy while reducing mean recurrent steps to 24.79 and slightly improving cross-entropy, showing that confidence-selected examples can safely omit the final frame; a modest threshold relaxation directly tests additional savings.

<<<<<<< SEARCH
        return confidence >= 0.97
=======
        return confidence >= 0.95
>>>>>>> REPLACE