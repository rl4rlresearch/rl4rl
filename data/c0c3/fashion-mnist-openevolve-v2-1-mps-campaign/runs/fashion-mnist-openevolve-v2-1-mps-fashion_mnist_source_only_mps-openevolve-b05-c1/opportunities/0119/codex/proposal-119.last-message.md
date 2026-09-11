MECHANISM: Fine-grained center-prior boundary search

HYPOTHESIS: A center weight of 1.8203125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860023609.

INTENDED_EDIT: Reduce the centered original and flipped logit weights from 1.828125 to 1.8203125 while preserving every evaluation view and the existing normalization.

EVIDENCE: Successive reductions from 1.875 to 1.84375 and then 1.828125 retained 9,360 correct while monotonically lowering cross-entropy; this half-step toward the unresolved 1.8125 candidate continues the verified direction with minimal classification-boundary risk.

<<<<<<< SEARCH
        center_weight = 1.828125
=======
        center_weight = 1.8203125
>>>>>>> REPLACE