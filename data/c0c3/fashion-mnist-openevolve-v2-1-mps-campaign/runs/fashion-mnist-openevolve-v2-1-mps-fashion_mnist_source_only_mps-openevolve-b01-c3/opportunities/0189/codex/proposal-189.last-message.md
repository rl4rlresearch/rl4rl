MECHANISM: Hybrid-pool temperature resharpening

HYPOTHESIS: Raising the hybrid ensemble’s calibration scale by 1% will preserve its 9,325 predictions while reducing validation cross-entropy below 0.193040839.

INTENDED_EDIT: Increase only the positive post-pooling logit multiplier from 1.22775 to 1.24003.

EVIDENCE: The 12.5% probability blend produced the best verified result but softens pooled probabilities; the prior identical calibration test timed out, providing no negative accuracy evidence, and positive scaling cannot change argmax predictions.

<<<<<<< SEARCH
        calibration = 1.22775 * (
=======
        calibration = 1.24003 * (
>>>>>>> REPLACE