MECHANISM: One-coordinate unflipped center-view refinement

HYPOTHESIS: Increasing only the unflipped center-view weight by one float32 ULP will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332153320314.

INTENDED_EDIT: Raise the unflipped center weight from 5.0498199462890625 to 5.049820423126221, leave the flipped center weight unchanged, and renormalize the ensemble denominator.

EVIDENCE: The combined first-step redistribution toward the unflipped center improved cross-entropy, while subsequently lowering only the flipped weight regressed to 0.21227332229614257; isolating the other component is the most informative unresolved center-weight direction.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0498199462890625, 5.049818992614746, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.09963893890381
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.049820423126221, 5.049818992614746, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.099639415740967
>>>>>>> REPLACE