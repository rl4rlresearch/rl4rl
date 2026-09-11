MECHANISM: Confidence-threshold boundary bisection

HYPOTHESIS: A 0.675 penultimate-step exit threshold will retain at least 85% validation accuracy while reducing recurrent steps and exact inference MACs versus the successful 0.70 design.

INTENDED_EDIT: Lower the final-frame early-exit confidence threshold from 0.70 to 0.675.

EVIDENCE: Threshold 0.70 passed at 85.03% accuracy, while 0.65 narrowly failed at 84.91%; testing their midpoint efficiently locates the lowest viable threshold.

<<<<<<< SEARCH
        return confidence >= 0.70
=======
        return confidence >= 0.675
>>>>>>> REPLACE