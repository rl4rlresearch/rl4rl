MECHANISM: One-ULP center-weight refinement

HYPOTHESIS: A 2.5249102115631104:1 center-to-shift ratio will preserve 9,254 correct predictions and reduce validation cross-entropy below 0.2122733253479004.

INTENDED_EDIT: Increase each center-view weight by one float32 ULP, from 5.0498199462890625 to 5.049820423126221, and renormalize the ensemble denominator.

EVIDENCE: The current ratio achieved the best cross-entropy, while the two-ULP-higher ratio regressed; the untested one-ULP midpoint is the smallest and most informative refinement between them.

<<<<<<< SEARCH
            (5.0498199462890625, 5.0498199462890625, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.099639892578125
=======
            (5.049820423126221, 5.049820423126221, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.09964084625244
>>>>>>> REPLACE