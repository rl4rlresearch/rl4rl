MECHANISM: Combined optimal crop fusion and reciprocal calibration

HYPOTHESIS: Using the verified 1.3515 crop-fusion power with the current reciprocal-multiplication calibration will preserve 9,349 correct predictions and reduce cross-entropy below 0.1876555103302002.

INTENDED_EDIT: Change only the crop-consensus power from 1.174 to 1.3515, retaining the current reciprocal temperature multiplier.

EVIDENCE: Reference Design 1 verified that power 1.3515 yields 9,349 correct and the best available cross-entropy; the current design separately verified that reciprocal multiplication slightly improves calibration at otherwise identical predictions.

<<<<<<< SEARCH
        fusion_power = 1.174
=======
        fusion_power = 1.3515
>>>>>>> REPLACE