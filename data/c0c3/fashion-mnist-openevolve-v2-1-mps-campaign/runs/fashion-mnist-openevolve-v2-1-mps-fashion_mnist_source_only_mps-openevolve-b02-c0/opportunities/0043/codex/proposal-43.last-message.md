MECHANISM: Fine-grained post-ensemble logit sharpening

HYPOTHESIS: Raising the ensemble scale from 1.05 to 1.06 will preserve exactly 9,252 correct predictions while lowering validation cross-entropy below 0.2287977.

INTENDED_EDIT: Increase the inference-only positive scale applied to the ten-view ensemble log-probabilities from 1.05 to 1.06.

EVIDENCE: Scaling from 1.00 to 1.05 preserved all 9,252 predictions and reduced cross-entropy from 0.2356299 to 0.2287977; the larger 1.075 and 1.10 trials timed out without contrary performance evidence, motivating a smaller continuation of the successful direction.

<<<<<<< SEARCH
        return 1.05 * ensemble_log_probabilities
=======
        return 1.06 * ensemble_log_probabilities
>>>>>>> REPLACE