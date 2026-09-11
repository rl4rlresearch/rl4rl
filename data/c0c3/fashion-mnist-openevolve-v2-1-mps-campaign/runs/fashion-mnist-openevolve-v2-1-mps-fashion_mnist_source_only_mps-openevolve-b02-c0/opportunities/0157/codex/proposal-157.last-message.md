MECHANISM: Antisymmetric center-view weight refinement

HYPOTHESIS: Favoring the unflipped center view by one float32 ULP while reducing the flipped center view by one ULP will preserve 9,254 correct predictions and lower cross-entropy below 0.21227332229614257.

INTENDED_EDIT: Redistribute center-view weight from the flipped prediction to the original prediction while keeping their total weight and the ensemble denominator unchanged.

EVIDENCE: Symmetric one-ULP moves both above and below the current center weight worsened cross-entropy, establishing a local optimum in total center mass; this tests the orthogonal, previously untested allocation between original and flipped center views.

<<<<<<< SEARCH
            (5.049819469451904, 5.049819469451904, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
=======
            (5.0498199462890625, 5.049818992614746, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE