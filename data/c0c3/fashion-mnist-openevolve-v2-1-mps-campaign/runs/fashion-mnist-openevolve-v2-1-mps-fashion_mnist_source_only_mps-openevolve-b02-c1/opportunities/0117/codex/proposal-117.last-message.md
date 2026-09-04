MECHANISM: Logit-ensemble temperature recalibration

HYPOTHESIS: Using unit temperature will preserve the weighted-logit ensemble’s 9,311 correct predictions while reducing validation cross-entropy below 0.192366.

INTENDED_EDIT: Remove the inherited 0.912 sharpening factor from evaluation logits without changing training, parameters, model-forward count, or class argmaxes.

EVIDENCE: Weighted logit averaging achieved the current best 9,311 correct; its 0.912 temperature came from probability aggregation, while the prior unit-temperature verification timed out and therefore did not test this computation-neutral recalibration.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.912)
=======
        return ensemble / 12.0
>>>>>>> REPLACE