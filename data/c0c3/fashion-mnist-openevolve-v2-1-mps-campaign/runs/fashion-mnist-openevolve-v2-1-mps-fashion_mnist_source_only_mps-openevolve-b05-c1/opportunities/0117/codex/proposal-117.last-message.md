MECHANISM: Fine-grained center-prior interpolation

HYPOTHESIS: A center weight of 1.84375 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860059734.

INTENDED_EDIT: Reduce the centered original and flipped logit weights from 1.875 to 1.84375 while preserving all views and normalization.

EVIDENCE: Reducing center weight from 2.0 to 1.875 retained 9,360 correct and lowered cross-entropy, while 1.5 lowered cross-entropy further but lost three predictions; this smaller step toward 1.5 targets calibration improvement with less boundary risk than the twice-timed-out 1.8125 candidate.

<<<<<<< SEARCH
        center_weight = 1.875
=======
        center_weight = 1.84375
>>>>>>> REPLACE