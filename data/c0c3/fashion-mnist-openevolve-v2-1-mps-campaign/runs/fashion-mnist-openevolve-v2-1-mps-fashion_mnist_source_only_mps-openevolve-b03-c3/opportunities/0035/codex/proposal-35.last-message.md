MECHANISM: Fine-grained post-ensemble confidence calibration

HYPOTHESIS: Scaling evaluation logits by 1.295 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.217585 by locating the calibration optimum between the strongly improving 1.25 scale and the nearly saturated 1.30 scale.

INTENDED_EDIT: Restore the proven training recipe and change its evaluation-only logit multiplier from 1.10 to 1.295.

EVIDENCE: Cross-entropy decreased monotonically from 0.218881 at 1.20 to 0.217864 at 1.25 and 0.217585 at 1.30 while all 9,240 predictions remained unchanged; the sharply diminishing gains indicate an optimum near 1.30, motivating a fine-grained interpolation rather than another 0.05 step.

<<<<<<< SEARCH
        return 1.10 * logits
=======
        return 1.295 * logits
>>>>>>> REPLACE