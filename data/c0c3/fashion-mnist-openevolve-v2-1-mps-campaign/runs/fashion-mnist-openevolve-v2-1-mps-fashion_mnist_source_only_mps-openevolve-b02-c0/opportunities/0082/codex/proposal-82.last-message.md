MECHANISM: Fine-grained arithmetic–geometric ensemble interpolation

HYPOTHESIS: Increasing the geometric contribution from 25% to 27.5% will retain at least 9,253 correct predictions while reducing validation cross-entropy below 0.2135743.

INTENDED_EDIT: Shift 2.5 percentage points of inference weight from arithmetic probability averaging to geometric log-probability averaging, retaining the verified training procedure, ten views, and 1.275 calibration.

EVIDENCE: The 25% blend achieved the best verified count of 9,253, while the fully geometric ensemble had lower cross-entropy but two fewer correct predictions; 27.5% conservatively probes the interval below the timed-out 30% attempt.

<<<<<<< SEARCH
        ensemble_scores = (
            0.75 * probability_ensemble + 0.25 * geometric_ensemble
        )
=======
        ensemble_scores = (
            0.725 * probability_ensemble + 0.275 * geometric_ensemble
        )
>>>>>>> REPLACE