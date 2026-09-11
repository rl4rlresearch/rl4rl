MECHANISM: Post-ensemble logit sharpening

HYPOTHESIS: Scaling ensemble log-probabilities by 1.10 will preserve exactly 9,252 correct predictions while lowering validation cross-entropy below 0.2287977.

INTENDED_EDIT: Increase the inference-only ensemble scale from 1.05 to 1.10.

EVIDENCE: Scaling from 1.00 to 1.05 preserved 9,252 correct predictions and reduced cross-entropy from 0.2356299 to 0.2287977; the prior 1.10 run timed out and therefore supplied no contrary performance evidence.

<<<<<<< SEARCH
        return 1.05 * ensemble_log_probabilities
=======
        return 1.10 * ensemble_log_probabilities
>>>>>>> REPLACE