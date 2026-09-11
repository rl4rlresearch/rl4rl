MECHANISM: Incremental post-ensemble confidence calibration

HYPOTHESIS: Increasing evaluation-logit scaling from 1.25 to 1.30 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217864.

INTENDED_EDIT: Restore the best verified 1.25 calibration and extend it by one proven 0.05 increment to 1.30, without changing training or predicted classes.

EVIDENCE: Scaling the same 9,240-correct model from 1.00 through 1.25 monotonically reduced cross-entropy at every 0.05 increment; the latest step improved it from 0.218881 to 0.217864 while preserving every argmax.

<<<<<<< SEARCH
        return 1.15 * logits
=======
        return 1.30 * logits
>>>>>>> REPLACE