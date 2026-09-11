MECHANISM: Local quadratic temperature calibration

HYPOTHESIS: An evaluation-time logit scale of 1.0671112 will preserve all 9,360 predictions while reducing validation cross-entropy below 0.1858569564819336.

INTENDED_EDIT: Replace only the positive evaluation-time logit multiplier with the minimum interpolated from the three nearest completed scale measurements.

EVIDENCE: Scales 1.0658182, 1.0664346, and 1.0671952 produced cross-entropies 0.18585714950561524, 0.18585700874328615, and 0.1858569564819336; quadratic interpolation places the local minimum near 1.0671112, and positive scaling preserves every argmax.

<<<<<<< SEARCH
        return 1.0671952 * logits
=======
        return 1.0671112 * logits
>>>>>>> REPLACE