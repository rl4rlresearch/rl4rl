MECHANISM: Fine-grained center-prior boundary search

HYPOTHESIS: A center weight of 1.828125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860035278.

INTENDED_EDIT: Reduce the centered original and flipped logit weights from 1.84375 to 1.828125 while preserving all views and normalization.

EVIDENCE: Lowering center weight from 1.875 to 1.84375 retained 9,360 correct and reduced cross-entropy from 0.1860059734 to 0.1860035278; 1.828125 continues that verified direction by only half the previously successful step, while the timed-out 1.8125 trials provide no contrary accuracy evidence.

<<<<<<< SEARCH
        center_weight = 1.84375
=======
        center_weight = 1.828125
>>>>>>> REPLACE