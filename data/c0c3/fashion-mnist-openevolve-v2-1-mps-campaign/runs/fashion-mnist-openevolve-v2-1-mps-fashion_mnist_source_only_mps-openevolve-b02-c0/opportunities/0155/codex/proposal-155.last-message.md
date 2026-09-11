MECHANISM: One-ULP downward center-weight refinement

HYPOTHESIS: Lowering each center-view weight by one float32 ULP will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.2122733253479004.

INTENDED_EDIT: Decrease both center-view weights from 5.0498199462890625 to 5.049819469451904 and renormalize the ensemble denominator to 26.09963893890381.

EVIDENCE: The current weight is the best verified point, while one and two ULP increases both regressed to 0.21227332611083985; testing the adjacent lower float is the most informative unresolved refinement.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0498199462890625, 5.0498199462890625, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.099639892578125
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.049819469451904, 5.049819469451904, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.09963893890381
>>>>>>> REPLACE