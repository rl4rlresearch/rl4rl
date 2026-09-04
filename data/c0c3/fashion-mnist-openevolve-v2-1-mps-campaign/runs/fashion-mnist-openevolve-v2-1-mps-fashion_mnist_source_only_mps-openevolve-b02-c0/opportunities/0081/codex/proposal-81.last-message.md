MECHANISM: Conservative geometric-ensemble interpolation

HYPOTHESIS: Increasing the geometric contribution from 25% to 30% will preserve at least 9,253 correct predictions while reducing validation cross-entropy below 0.2135743.

INTENDED_EDIT: Shift five percentage points of inference weight from arithmetic probability averaging to geometric log-probability averaging, leaving training and calibration unchanged.

EVIDENCE: The 25% geometric blend improved validation_correct from 9,252 to 9,253, while the fully geometric ensemble had lower cross-entropy but lost two predictions; a small increase tests whether more of its calibration benefit is available within the current accuracy plateau.

<<<<<<< SEARCH
        ensemble_scores = (
            0.75 * probability_ensemble + 0.25 * geometric_ensemble
        )
=======
        ensemble_scores = (
            0.70 * probability_ensemble + 0.30 * geometric_ensemble
        )
>>>>>>> REPLACE