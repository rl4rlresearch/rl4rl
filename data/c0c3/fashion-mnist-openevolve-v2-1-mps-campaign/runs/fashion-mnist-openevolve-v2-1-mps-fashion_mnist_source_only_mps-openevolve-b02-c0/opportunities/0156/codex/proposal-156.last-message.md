MECHANISM: One-ULP downward center-weight refinement

HYPOTHESIS: Lowering each center-view weight by one additional float32 ULP will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.21227332229614257.

INTENDED_EDIT: Decrease both center-view weights from 5.049819469451904 to 5.049818992614746 and renormalize the ensemble denominator to 26.099637985229492.

EVIDENCE: The previous one-ULP decrease improved cross-entropy from 0.2122733253479004 to 0.21227332229614257 while preserving all 9,254 correct predictions, whereas increases regressed; continuing one adjacent float32 step downward is the most informative local refinement.

<<<<<<< SEARCH
            (5.049819469451904, 5.049819469451904, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.09963893890381
=======
            (5.049818992614746, 5.049818992614746, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.099637985229492
>>>>>>> REPLACE