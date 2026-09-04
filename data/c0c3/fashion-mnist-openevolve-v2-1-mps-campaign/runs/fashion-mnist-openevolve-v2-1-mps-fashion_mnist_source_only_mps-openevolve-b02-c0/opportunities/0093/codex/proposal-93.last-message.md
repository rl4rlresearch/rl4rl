MECHANISM: Intermediate post-ensemble temperature sharpening

HYPOTHESIS: Scaling ensemble scores from 1.281 to 1.32 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.21344894.

INTENDED_EDIT: Change only the positive inference-time calibration multiplier; retain the verified architecture, training procedure, views, and ensemble weights.

EVIDENCE: Every verified scale increase through 1.281 preserved 9,253 correct predictions and monotonically reduced cross-entropy. A positive multiplier preserves argmax predictions, while 1.32 probes toward the previously estimated optimum near 1.36 without repeating a timed-out setting.

<<<<<<< SEARCH
        return 1.281 * ensemble_scores
=======
        return 1.32 * ensemble_scores
>>>>>>> REPLACE