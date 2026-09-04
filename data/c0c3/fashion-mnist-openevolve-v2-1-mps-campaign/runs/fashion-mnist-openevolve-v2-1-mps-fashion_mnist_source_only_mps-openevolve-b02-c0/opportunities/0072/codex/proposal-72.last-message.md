MECHANISM: Post-ensemble calibration sharpening

HYPOTHESIS: Increasing the positive ensemble scale from 1.10 to 1.20 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2224249.

INTENDED_EDIT: Retain the verified arithmetic–geometric ensemble and change only its inference-time calibration scale.

EVIDENCE: The current 25% geometric blend achieved the best verified count of 9,253; positive scaling cannot change its argmax predictions, and the prior 1.05-to-1.10 increase reduced cross-entropy from 0.2287977 to 0.2235582 without changing predictions.

<<<<<<< SEARCH
        return 1.10 * ensemble_scores
=======
        return 1.20 * ensemble_scores
>>>>>>> REPLACE