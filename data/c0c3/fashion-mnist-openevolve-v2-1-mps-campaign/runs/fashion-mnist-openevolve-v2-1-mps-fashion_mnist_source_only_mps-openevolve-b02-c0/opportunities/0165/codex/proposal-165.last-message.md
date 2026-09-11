MECHANISM: One-coordinate unflipped center-view refinement

HYPOTHESIS: Increasing only the unflipped center-view weight by one float32 ULP will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332153320314.

INTENDED_EDIT: Raise the unflipped center weight to its next float32 value, leave all other view weights unchanged, and renormalize the ensemble denominator.

EVIDENCE: The successful antisymmetric redistribution favored the unflipped center view, while lowering only the flipped weight regressed; this complementary one-coordinate change remains unresolved because its sole verification timed out.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0498199462890625, 5.049818992614746, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.09963893890381
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.049820423126221, 5.049818992614746, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.099639415740967
>>>>>>> REPLACE