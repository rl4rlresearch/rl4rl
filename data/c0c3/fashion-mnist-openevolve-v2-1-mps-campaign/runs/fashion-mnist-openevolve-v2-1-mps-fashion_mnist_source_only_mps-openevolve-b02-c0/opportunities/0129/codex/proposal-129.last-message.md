MECHANISM: Intermediate center-weight extrapolation in test-time augmentation

HYPOTHESIS: Increasing the center-to-shift view-weight ratio from 2.5:1 to 2.75:1 will exceed 9,254 correct predictions while preserving more useful shifted-view averaging than the repeatedly timed-out 3:1 proposal.

INTENDED_EDIT: Increase both center-view weights from 5.0 to 5.5 and renormalize the ten-view ensemble from 26.0 to 27.0, retaining the probability/geometric blend and calibration.

EVIDENCE: Moving from uniform weights to a 2.5:1 center emphasis improved validation_correct from 9,253 to 9,254; 2.75:1 is the most informative untested interpolation toward the twice-unresolved 3:1 design.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.0
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.5, 5.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 27.0
>>>>>>> REPLACE