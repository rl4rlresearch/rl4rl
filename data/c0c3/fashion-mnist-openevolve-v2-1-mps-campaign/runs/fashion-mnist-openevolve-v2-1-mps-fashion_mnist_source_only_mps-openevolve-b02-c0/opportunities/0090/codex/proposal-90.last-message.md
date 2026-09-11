MECHANISM: Fine-grained post-ensemble temperature sharpening

HYPOTHESIS: Increasing the ensemble scale from 1.281 to 1.28125 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.21344894.

INTENDED_EDIT: Change only the positive inference-time calibration multiplier, retaining the verified architecture, training procedure, views, and ensemble weights.

EVIDENCE: The verified increase from 1.280 to 1.281 preserved exactly 9,253 correct predictions and reduced cross-entropy from 0.21346918 to 0.21344894. Positive scaling preserves argmax decisions, and the timed-out 1.282 attempt supplied no contrary validation evidence.

<<<<<<< SEARCH
        return 1.281 * ensemble_scores
=======
        return 1.28125 * ensemble_scores
>>>>>>> REPLACE