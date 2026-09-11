MECHANISM: One-coordinate center-view refinement

HYPOTHESIS: Lowering only the flipped center-view weight by one float32 ULP will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332153320314.

INTENDED_EDIT: Decrease the flipped center weight from 5.049818992614746 to 5.049818515777588 while leaving the unflipped weight unchanged and renormalizing the ensemble.

EVIDENCE: The first antisymmetric redistribution toward the unflipped center improved cross-entropy, its second step tied, and symmetric downward movement caused less regression than symmetric upward movement; isolating the downward flipped-weight component tests the most promising unresolved intermediate direction.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0498199462890625, 5.049818992614746, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.09963893890381
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0498199462890625, 5.049818515777588, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.09963846206665
>>>>>>> REPLACE