MECHANISM: Conservative center-weight extrapolation in test-time augmentation

HYPOTHESIS: Increasing the center-to-shift view weight ratio from 2.5:1 to 3:1 will exceed 9,254 correct predictions by further reducing translation-induced distortion while retaining useful shifted-view averaging.

INTENDED_EDIT: Increase each center-view ensemble weight from 5/26 to 6/28 and renormalize all ten inference weights; retain the existing probability/geometric blend and calibration.

EVIDENCE: Moving from uniform offset weights to the training-prior 2.5:1 center weighting improved validation_correct from 9,253 to 9,254, providing directional evidence that moderately greater center emphasis is beneficial.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.0
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (6.0, 6.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 28.0
>>>>>>> REPLACE