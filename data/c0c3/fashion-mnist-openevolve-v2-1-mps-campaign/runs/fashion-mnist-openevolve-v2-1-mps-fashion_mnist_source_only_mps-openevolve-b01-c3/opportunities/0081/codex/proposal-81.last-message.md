MECHANISM: Decision-invariant temperature calibration

HYPOTHESIS: Scaling the pooled logits by 1.175 will preserve all 9,287 argmax predictions while reducing validation cross-entropy below 0.2085474.

INTENDED_EDIT: Increase only the final ten-view evaluation logit scale from 1.125 to 1.175.

EVIDENCE: Increasing the scale from 1.10 to 1.125 preserved 9,287 correct predictions and reduced cross-entropy from 0.2096186 to 0.2085474; positive scaling cannot change argmax predictions, and the timed-out 1.15 and 1.20 trials provide no subject-level evidence against further sharpening.

<<<<<<< SEARCH
        return 1.125 * view_logits.mean(dim=(0, 1))
=======
        return 1.175 * view_logits.mean(dim=(0, 1))
>>>>>>> REPLACE