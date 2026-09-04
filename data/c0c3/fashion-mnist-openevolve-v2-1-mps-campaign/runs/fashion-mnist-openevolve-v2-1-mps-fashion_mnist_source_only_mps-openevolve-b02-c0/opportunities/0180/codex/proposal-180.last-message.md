MECHANISM: Unflipped-view concentrated vertical-shift weighting

HYPOTHESIS: Moving the successful vertical redistribution from flipped views into unflipped views will preserve 9,254 correct predictions and reduce cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Restore the flipped vertical-shift weights to 2.0 and double the existing redistribution in their unflipped counterparts, preserving aggregate spatial weights and the denominator.

EVIDENCE: Concentrating the redistribution in flipped views regressed cross-entropy to 0.21227332153320314; testing the exact opposite orientation redistribution is the most informative remaining local direction.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 1.999999761581421, 2.0, 2.0, 2.0, 2.0, 2.000000238418579, 2.000000238418579)
=======
            (5.0498199462890625, 5.049818992614746, 1.9999995231628418, 2.0, 2.0, 2.0, 2.0, 2.0, 2.000000476837158, 2.0)
>>>>>>> REPLACE