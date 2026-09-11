MECHANISM: Antisymmetric vertical-shift view weighting

HYPOTHESIS: Transferring one effective float32 step of weight from the final vertical-shift pair to the first vertical-shift pair will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332153320314.

INTENDED_EDIT: Increase both first vertical-shift weights to 2.000000238418579 and decrease both opposing vertical-shift weights to 1.999999761581421, preserving total ensemble weight and the denominator.

EVIDENCE: The identical vertical-shift direction was the most recent evidence-grounded proposal but timed out, so its effect remains unresolved; center and orientation-weight refinements have already plateaued or regressed.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
=======
            (5.0498199462890625, 5.049818992614746, 2.000000238418579, 2.000000238418579, 2.0, 2.0, 2.0, 2.0, 1.999999761581421, 1.999999761581421)
>>>>>>> REPLACE