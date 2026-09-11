MECHANISM: Relaxed confidence-gated final-frame skipping

HYPOTHESIS: Lowering the penultimate-step exit threshold from 0.95 to 0.90 will preserve at least 85% validation accuracy while allowing more examples to skip the 25th recurrent step, reducing exact inference MACs and mean recurrent steps.

INTENDED_EDIT: Exit after 24 of 25 scheduled steps when provisional classifier confidence is at least 0.90 instead of 0.95.

EVIDENCE: Lowering the threshold from 0.97 to 0.95 increased final-frame skipping, reduced mean recurrent steps from 24.790 to 24.720, and improved validation accuracy from 85.03% to 85.15%; this supports testing a further moderate relaxation.

<<<<<<< SEARCH
        return confidence >= 0.95
=======
        return confidence >= 0.90
>>>>>>> REPLACE