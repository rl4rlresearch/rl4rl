MECHANISM: Incremental post-ensemble confidence calibration

HYPOTHESIS: Increasing evaluation-logit scaling from 1.20 to 1.25 will preserve all 9,240 argmax predictions and reduce validation cross-entropy below 0.218881.

INTENDED_EDIT: Increase the positive evaluation-only logit multiplier to 1.25 without changing training or predicted classes.

EVIDENCE: Scaling the same 9,240-correct model from 1.00 through 1.20 monotonically reduced cross-entropy at every 0.05 increment, with the latest change improving it from 0.220795 to 0.218881 while preserving every argmax.

<<<<<<< SEARCH
        return 1.20 * logits
=======
        return 1.25 * logits
>>>>>>> REPLACE