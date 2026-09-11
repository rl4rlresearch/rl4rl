MECHANISM: Reverse antisymmetric vertical-shift view weighting

HYPOTHESIS: Transferring one float32 weight step from both orientations of the first vertical crop to the opposing vertical crop will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332153320314.

INTENDED_EDIT: Decrease both first vertical-shift weights to 1.999999761581421 and increase both opposing vertical-shift weights to 2.000000238418579, preserving total ensemble weight.

EVIDENCE: The forward vertical redistribution preserved 9,254 correct predictions but worsened cross-entropy to 0.21227332458496093; its exact reverse is the strongest local direction, and the prior reverse test timed out without resolving its effect.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
=======
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000238418579, 2.000000238418579)
>>>>>>> REPLACE