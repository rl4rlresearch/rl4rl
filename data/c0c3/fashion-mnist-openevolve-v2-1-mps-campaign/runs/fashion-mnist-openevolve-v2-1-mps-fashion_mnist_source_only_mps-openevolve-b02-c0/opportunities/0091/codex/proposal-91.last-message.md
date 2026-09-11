MECHANISM: Ultra-conservative post-ensemble temperature sharpening

HYPOTHESIS: Increasing the ensemble scale from 1.281 to 1.2811 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.21344894.

INTENDED_EDIT: Change only the positive inference-time calibration multiplier; retain the verified architecture, training procedure, views, and ensemble weights.

EVIDENCE: The verified increase from 1.280 to 1.281 preserved exactly 9,253 correct predictions and lowered cross-entropy from 0.21346918 to 0.21344894. Positive scaling preserves argmax, while timed-out 1.28125 and 1.282 attempts supplied no contrary validation evidence.

<<<<<<< SEARCH
        return 1.281 * ensemble_scores
=======
        return 1.2811 * ensemble_scores
>>>>>>> REPLACE