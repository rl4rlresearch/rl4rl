MECHANISM: One-ULP center-weight refinement

HYPOTHESIS: Raising each center-view weight by one float32 ULP will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.2122733253479004.

INTENDED_EDIT: Increase both center-view weights from 5.0498199462890625 to 5.049820423126221 and renormalize the ensemble denominator to 26.09964084625244.

EVIDENCE: The current weight is the best verified point, while the two-ULP-higher weight regressed; the intervening one-ULP candidate remains unresolved because its previous verifications timed out without contrary validation evidence.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0498199462890625, 5.0498199462890625, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.099639892578125
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.049820423126221, 5.049820423126221, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.09964084625244
>>>>>>> REPLACE