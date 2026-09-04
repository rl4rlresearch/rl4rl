MECHANISM: Reverse orientation-wide shifted-view weighting

HYPOTHESIS: Favoring flipped shifted views by the same increment that previously favored unflipped shifted views will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332153320314.

INTENDED_EDIT: Transfer equal weight from each unflipped shifted view to its flipped counterpart while preserving every shifted pair’s total weight and the ensemble denominator.

EVIDENCE: The opposite orientation-wide redistribution preserved 9,254 correct predictions but worsened cross-entropy to 0.21227332191467285, providing direct local evidence that moving shifted-view weight in the reverse direction is the most informative next test.

<<<<<<< SEARCH
            (5.0498199462890625, 5.049818992614746, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
=======
            (5.0498199462890625, 5.049818992614746, 1.999999761581421, 2.000000238418579, 1.999999761581421, 2.000000238418579, 1.999999761581421, 2.000000238418579, 1.999999761581421, 2.000000238418579)
>>>>>>> REPLACE