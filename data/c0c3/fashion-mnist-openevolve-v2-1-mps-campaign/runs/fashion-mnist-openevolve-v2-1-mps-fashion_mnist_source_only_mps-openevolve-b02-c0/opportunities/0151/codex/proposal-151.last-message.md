MECHANISM: One-ULP center-weight refinement

HYPOTHESIS: A 2.5249102115631104:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.2122733253479004.

INTENDED_EDIT: Raise both center-view weights by one float32 ULP and renormalize the ten-view ensemble.

EVIDENCE: The current weight is the best verified point, while the two-ULP-higher weight regressed; the intervening one-ULP design remains the most informative unresolved candidate because its prior verifications timed out without producing contrary results.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0498199462890625, 5.0498199462890625, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.099639892578125
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.049820423126221, 5.049820423126221, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.09964084625244
>>>>>>> REPLACE