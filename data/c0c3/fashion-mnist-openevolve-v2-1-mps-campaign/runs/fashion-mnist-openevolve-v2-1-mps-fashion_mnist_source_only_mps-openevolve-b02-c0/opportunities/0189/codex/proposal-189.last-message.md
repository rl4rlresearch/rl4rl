MECHANISM: Center-view flip-orientation weight separation

HYPOTHESIS: Increasing the existing center-view orientation imbalance by one float32 step per orientation will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Transfer one float32 weight step from the flipped center view to the higher-weight unflipped center view while preserving their combined weight and the ensemble denominator.

EVIDENCE: Equalizing these center-view weights worsened cross-entropy to 0.2122733253479004 without changing correctness; the opposite neighboring redistribution remains unresolved because its verification timed out.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000238418579, 2.000000238418579)
=======
            (5.049820423126221, 5.049818515777588, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000238418579, 2.000000238418579)
>>>>>>> REPLACE