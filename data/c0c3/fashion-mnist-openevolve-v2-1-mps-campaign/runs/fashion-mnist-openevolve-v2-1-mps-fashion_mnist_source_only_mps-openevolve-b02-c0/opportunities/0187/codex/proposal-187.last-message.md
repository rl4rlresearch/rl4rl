MECHANISM: Center-view flip-orientation weight equalization

HYPOTHESIS: Equalizing the two center-view weights will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Transfer one float32 weight step from the higher-weight unflipped center view to its flipped counterpart, preserving their sum and the ensemble denominator.

EVIDENCE: Both directions of the tested logit-scale and probability–geometric refinements regressed while preserving correctness; the center-view orientation imbalance is a remaining independent local ensemble coordinate.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000238418579, 2.000000238418579)
=======
            (5.049819469451904, 5.049819469451904, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000238418579, 2.000000238418579)
>>>>>>> REPLACE