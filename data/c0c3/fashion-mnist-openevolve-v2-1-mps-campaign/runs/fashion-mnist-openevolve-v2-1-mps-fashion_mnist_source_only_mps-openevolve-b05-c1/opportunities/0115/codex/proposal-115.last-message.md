MECHANISM: Conservative center-prior refinement

HYPOTHESIS: A center weight of 1.8125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860059734.

INTENDED_EDIT: Reduce the centered original and flipped logit weights from 1.875 to 1.8125, preserving all views and normalization.

EVIDENCE: Weight 1.875 retained the 9,360 correct predictions of weight 2.0 while lowering cross-entropy, whereas weight 1.5 lowered cross-entropy further but lost three correct; a half-step toward 1.5 tests additional calibration benefit conservatively.

<<<<<<< SEARCH
        center_weight = 1.875
=======
        center_weight = 1.8125
>>>>>>> REPLACE