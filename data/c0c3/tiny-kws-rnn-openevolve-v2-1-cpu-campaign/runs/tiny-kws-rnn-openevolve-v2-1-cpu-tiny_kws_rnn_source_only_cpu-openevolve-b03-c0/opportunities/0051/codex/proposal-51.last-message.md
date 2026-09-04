MECHANISM: Confidence-threshold boundary bisection

HYPOTHESIS: A 0.6875 penultimate-step exit threshold will retain at least 85% validation accuracy while reducing recurrent steps and exact inference MACs versus the successful 0.70 design.

INTENDED_EDIT: Lower the final-frame early-exit confidence threshold from 0.70 to 0.6875.

EVIDENCE: Threshold 0.70 passed at 85.03% accuracy, while 0.675 narrowly failed at 84.91%; their midpoint is the most informative next test of the viability boundary.

<<<<<<< SEARCH
        return confidence >= 0.70
=======
        return confidence >= 0.6875
>>>>>>> REPLACE