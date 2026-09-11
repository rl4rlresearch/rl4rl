MECHANISM: Symmetric vertical-shift weight extrapolation

HYPOTHESIS: Transferring 2.384185791015625e-7 per orientation from the first vertical crop to its opposing crop will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Decrease both first-vertical-crop ensemble weights and equally increase both opposing-vertical-crop weights, preserving total ensemble weight and denominator.

EVIDENCE: The current symmetric vertical redistribution produced the best verified cross-entropy, while orientation-specific redistributions regressed; its direct symmetric extrapolation remains unresolved because prior implementations could not be verified.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000238418579, 2.000000238418579)
=======
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 1.999999761581421, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.000000476837158, 2.000000476837158)
>>>>>>> REPLACE