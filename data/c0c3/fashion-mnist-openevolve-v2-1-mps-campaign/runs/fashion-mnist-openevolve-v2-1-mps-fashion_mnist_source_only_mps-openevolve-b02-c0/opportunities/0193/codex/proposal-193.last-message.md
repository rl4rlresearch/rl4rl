MECHANISM: Symmetric vertical-shift weight extrapolation

HYPOTHESIS: Extending the successful paired vertical redistribution by 2.384185791015625e-7 per orientation will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Transfer equal weight from both orientations of the first vertical crop to both orientations of the opposing vertical crop while preserving total ensemble weight and denominator.

EVIDENCE: The current symmetric vertical weighting produced the best measured cross-entropy, while orientation-specific redistributions regressed; its direct extrapolation remains unresolved because prior attempts could not be verified.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000238418579, 2.000000238418579)
=======
            (5.0498199462890625, 5.049818992614746, 1.9999995231628418, 1.9999995231628418, 2.0, 2.0, 2.0, 2.0, 2.000000476837158, 2.000000476837158)
>>>>>>> REPLACE