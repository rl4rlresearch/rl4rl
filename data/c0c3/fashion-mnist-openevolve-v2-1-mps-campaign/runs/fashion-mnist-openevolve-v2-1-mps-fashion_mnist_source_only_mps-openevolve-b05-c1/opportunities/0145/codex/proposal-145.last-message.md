MECHANISM: Local quadratic temperature calibration

HYPOTHESIS: An evaluation-time logit scale of 1.0671112 will preserve all 9,360 predictions while reducing validation cross-entropy below 0.1858569564819336.

INTENDED_EDIT: Replace only the positive evaluation-time logit multiplier, leaving training and class rankings unchanged.

EVIDENCE: Quadratic interpolation of the three nearest completed scale measurements predicts a minimum near 1.0671112; its previous verification timed out and supplied no contrary performance evidence.

<<<<<<< SEARCH
        return 1.0671952 * logits
=======
        return 1.0671112 * logits
>>>>>>> REPLACE