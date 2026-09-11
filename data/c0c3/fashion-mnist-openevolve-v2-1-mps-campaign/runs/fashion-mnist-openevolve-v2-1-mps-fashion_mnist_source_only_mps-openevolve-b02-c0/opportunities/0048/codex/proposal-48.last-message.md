MECHANISM: Post-ensemble logit sharpening

HYPOTHESIS: Increasing the positive ensemble scale from 1.10 to 1.20 will preserve exactly 9,252 correct predictions while lowering validation cross-entropy below 0.2235582.

INTENDED_EDIT: Raise the inference-only scale applied to the unchanged ten-view ensemble log-probabilities from 1.10 to 1.20.

EVIDENCE: With pair-batched evaluation, increasing the scale from 1.05 to 1.10 preserved all 9,252 predictions and reduced cross-entropy from 0.2287977 to 0.2235582; positive scaling cannot change the ensemble argmax, and the continued cross-entropy improvement indicates remaining underconfidence.

<<<<<<< SEARCH
        return 1.10 * ensemble_log_probabilities
=======
        return 1.20 * ensemble_log_probabilities
>>>>>>> REPLACE