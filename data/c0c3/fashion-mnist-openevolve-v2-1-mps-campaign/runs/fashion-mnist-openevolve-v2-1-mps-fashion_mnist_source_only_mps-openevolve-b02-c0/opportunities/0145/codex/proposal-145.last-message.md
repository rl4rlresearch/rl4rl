MECHANISM: Minimum-step center-weight refinement

HYPOTHESIS: A 2.524913787841797:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.2122733253479004.

INTENDED_EDIT: Increase both center-view weights from 5.0498199462890625 to 5.049827575683594 and renormalize the ten-view ensemble from 26.099639892578125 to 26.099655151367188.

EVIDENCE: Cross-entropy improved monotonically through ratios 2.5234375:1, 2.5244140625:1, 2.52490234375:1, and 2.52490997314453125:1 while retaining 9,254 correct; this midpoint toward the timed-out 2.5249176025390625:1 trial is the smallest informative continuation.

<<<<<<< SEARCH
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0498199462890625, 5.0498199462890625, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.099639892578125
=======
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.049827575683594, 5.049827575683594, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.099655151367188
>>>>>>> REPLACE