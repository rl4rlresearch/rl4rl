MECHANISM: Orientation-selective half-step vertical interpolation

HYPOTHESIS: Applying the unresolved second vertical redistribution only to unflipped views will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Decrease the unflipped first vertical-crop weight by one float32 step and increase the unflipped opposing-crop weight by one float32 step, leaving flipped weights and total ensemble weight unchanged.

EVIDENCE: The first full reverse vertical redistribution improved cross-entropy, while the second full step regressed; this orientation-selective midpoint is the most informative untested setting because its previous verification timed out.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000238418579, 2.000000238418579)
=======
            (5.0498199462890625, 5.049818992614746, 1.9999995231628418, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000476837158, 2.000000238418579)
>>>>>>> REPLACE