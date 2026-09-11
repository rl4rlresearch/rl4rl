MECHANISM: Fine-grained center-weight extrapolation in test-time augmentation

HYPOTHESIS: Increasing the center-to-shift ratio from 2.5:1 to 2.53125:1 will exceed 9,254 correct predictions while retaining more useful shifted-view information than the unresolved higher-ratio designs.

INTENDED_EDIT: Increase both center-view weights from 5.0 to 5.0625 and renormalize the ten-view ensemble from 26.0 to 26.125.

EVIDENCE: Moving from uniform weighting to 2.5:1 center emphasis improved validation_correct from 9,253 to 9,254; all larger extrapolations timed out without contradictory validation evidence, motivating a smaller untested step in the supported direction.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.0
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0625, 5.0625, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.125
>>>>>>> REPLACE