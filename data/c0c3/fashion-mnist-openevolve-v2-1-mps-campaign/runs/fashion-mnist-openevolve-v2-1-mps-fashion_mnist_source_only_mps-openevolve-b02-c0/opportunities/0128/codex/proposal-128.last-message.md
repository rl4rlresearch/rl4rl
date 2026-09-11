MECHANISM: Conservative center-weight extrapolation in test-time augmentation

HYPOTHESIS: Increasing the center-to-shift view-weight ratio from 2.5:1 to 3:1 will exceed 9,254 correct predictions by further reducing translation-induced distortion while preserving useful shifted-view averaging.

INTENDED_EDIT: Increase both center-view weights from 5 to 6 and renormalize the ten-view ensemble from 26 to 28, retaining the probability/geometric blend and calibration.

EVIDENCE: Moving from uniform weighting to a 2.5:1 center emphasis improved validation_correct from 9,253 to 9,254; the sole 3:1 verification timed out and provided no contradictory validation evidence.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.0
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (6.0, 6.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 28.0
>>>>>>> REPLACE