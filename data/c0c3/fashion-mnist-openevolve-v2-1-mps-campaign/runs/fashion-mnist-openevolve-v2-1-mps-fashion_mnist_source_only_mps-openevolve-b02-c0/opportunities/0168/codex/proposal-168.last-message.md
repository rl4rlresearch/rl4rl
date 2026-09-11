MECHANISM: Antisymmetric vertical-shift view weighting

HYPOTHESIS: Transferring weight from the final vertical-shift pair to the first vertical-shift pair will preserve 9,254 correct predictions and lower validation cross-entropy below 0.21227332153320314.

INTENDED_EDIT: Increase both orientations of the first vertical crop and decrease both orientations of the opposing vertical crop by the same numeric amount, preserving total ensemble weight and the denominator.

EVIDENCE: Center-view and orientation-wide directions have reached local regressions or plateaus; allocation between opposing spatial shifts is an orthogonal untested direction, and paired changes preserve the currently successful orientation balance.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0498199462890625, 5.049818992614746, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.09963893890381
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0498199462890625, 5.049818992614746, 2.000000238418579, 2.000000238418579, 2.0, 2.0, 2.0, 2.0, 1.999999761581421, 1.999999761581421)
        ) / 26.09963893890381
>>>>>>> REPLACE