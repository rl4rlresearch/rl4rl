MECHANISM: Opposite antisymmetric horizontal-shift view weighting

HYPOTHESIS: Increasing both first horizontal-crop weights by one float32 step while decreasing both opposing horizontal-crop weights equally will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Transfer one effective float32 weight step from both orientations of the opposing horizontal crop to the first horizontal crop, preserving total ensemble weight and denominator.

EVIDENCE: The reverse horizontal redistribution is the most informative unresolved direction: the opposite change regressed cross-entropy to 0.21227332229614257, while its prior verification timed out without measuring its effect.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000238418579, 2.000000238418579)
=======
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 1.999999761581421, 2.000000238418579, 2.000000238418579, 1.999999761581421, 1.999999761581421, 2.000000238418579, 2.000000238418579)
>>>>>>> REPLACE