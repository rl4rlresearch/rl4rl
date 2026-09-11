MECHANISM: Orientation-redistributed vertical-shift weighting

HYPOTHESIS: Concentrating the successful vertical redistribution in flipped views will preserve 9,254 correct predictions and lower cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Restore the unflipped vertical-shift weights to 2.0 and move their redistribution to the flipped counterparts, preserving every aggregate spatial weight and the denominator.

EVIDENCE: Extending the redistribution only in flipped views produced lower cross-entropy (0.21227332229614257) than extending it only in unflipped views (0.21227332305908203), suggesting the flipped orientation better tolerates additional vertical redistribution.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000238418579, 2.000000238418579)
=======
            (5.0498199462890625, 5.049818992614746, 2.0, 1.9999995231628418, 2.0, 2.0, 2.0, 2.0, 2.0, 2.000000476837158)
>>>>>>> REPLACE