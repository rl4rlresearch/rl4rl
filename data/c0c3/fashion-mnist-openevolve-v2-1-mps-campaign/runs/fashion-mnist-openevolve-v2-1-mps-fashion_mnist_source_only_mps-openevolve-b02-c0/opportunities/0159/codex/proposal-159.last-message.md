MECHANISM: Second-step antisymmetric center-view weight refinement

HYPOTHESIS: Extending the successful redistribution by one additional float32 ULP will preserve 9,254 correct predictions and lower cross-entropy below 0.21227332153320314.

INTENDED_EDIT: Increase the unflipped center-view weight by one float32 ULP and decrease the flipped center-view weight by one ULP, preserving total center mass and the ensemble denominator.

EVIDENCE: The first antisymmetric one-ULP redistribution improved cross-entropy from 0.21227332229614257 to 0.21227332153320314 with all 9,254 correct predictions preserved; the second step remains unresolved because its only verification timed out.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
=======
            (5.049820423126221, 5.049818515777588, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE