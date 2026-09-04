MECHANISM: Minimum-step center-weight refinement

HYPOTHESIS: A 2.5249176025390625:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332611083985.

INTENDED_EDIT: Increase both center-view weights from 5.0498046875 to 5.049835205078125 and renormalize the ten-view ensemble from 26.099609375 to 26.09967041015625.

EVIDENCE: Cross-entropy improved monotonically through ratios 2.5234375:1, 2.5244140625:1, and 2.52490234375:1 while retaining 9,254 correct; this midpoint toward the timed-out 2.524932861328125:1 trial is the smallest untested continuation in the supported direction.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0498046875, 5.0498046875, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.099609375
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.049835205078125, 5.049835205078125, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.09967041015625
>>>>>>> REPLACE