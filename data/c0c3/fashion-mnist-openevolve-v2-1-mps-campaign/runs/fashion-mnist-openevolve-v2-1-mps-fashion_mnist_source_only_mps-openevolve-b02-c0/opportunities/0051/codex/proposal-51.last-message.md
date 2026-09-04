MECHANISM: Midpoint post-ensemble logit sharpening

HYPOTHESIS: Raising the scale from 1.10 to 1.15 will preserve exactly 9,252 correct predictions while reducing validation cross-entropy below 0.2235582.

INTENDED_EDIT: Increase only the positive scale applied to the unchanged pair-batched ten-view ensemble log-probabilities.

EVIDENCE: Scaling from 1.05 to 1.10 preserved all 9,252 predictions and lowered cross-entropy from 0.2287977 to 0.2235582; positive scaling cannot alter argmax, while 1.15 with the successful pair-batched evaluation path remains untested.

<<<<<<< SEARCH
        return 1.10 * ensemble_log_probabilities
=======
        return 1.15 * ensemble_log_probabilities
>>>>>>> REPLACE