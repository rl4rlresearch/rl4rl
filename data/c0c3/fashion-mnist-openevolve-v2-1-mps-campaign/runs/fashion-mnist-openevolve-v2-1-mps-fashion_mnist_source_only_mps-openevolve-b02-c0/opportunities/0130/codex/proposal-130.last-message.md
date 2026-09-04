MECHANISM: Local center-weight interpolation in test-time augmentation

HYPOTHESIS: Increasing the center-to-shift weight ratio from 2.5:1 to 2.625:1 will exceed 9,254 correct predictions while retaining more shifted-view information than the unresolved 2.75:1 and 3:1 designs.

INTENDED_EDIT: Increase both center-view weights from 5.0 to 5.25 and renormalize the ten-view ensemble from 26.0 to 26.5.

EVIDENCE: Increasing center emphasis from uniform weighting to 2.5:1 improved validation_correct from 9,253 to 9,254; the higher-ratio attempts timed out without contradictory score evidence, motivating a smaller interpolation beyond the verified optimum.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.0
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.25, 5.25, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.5
>>>>>>> REPLACE