MECHANISM: Orientation-selective half-step vertical interpolation

HYPOTHESIS: Applying the next vertical redistribution only to the unflipped views will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Decrease the unflipped first-vertical-crop weight to 1.9999995231628418 and increase the unflipped opposing-crop weight to 2.000000476837158, preserving total ensemble weight while leaving their flipped counterparts unchanged.

EVIDENCE: The first full vertical redistribution improved cross-entropy, but the second full step regressed to 0.2122733238220215; changing one orientation implements the informative aggregate midpoint between those two tested settings.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000238418579, 2.000000238418579)
=======
            (5.0498199462890625, 5.049818992614746, 1.9999995231628418, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000476837158, 2.000000238418579)
>>>>>>> REPLACE