MECHANISM: Combined optimal crop fusion and interpolated temperature calibration

HYPOTHESIS: Using fusion power 1.3515 with the current interpolated calibration will preserve 9,349 correct predictions and reduce cross-entropy below 0.18765550994873048.

INTENDED_EDIT: Change only the crop-consensus fusion power from 1.174 to the best verified value, retaining the current finer temperature interpolation.

EVIDENCE: Fusion power 1.3515 achieved 9,349 correct with the best available cross-entropy, while the current interpolation reduced cross-entropy at power 1.174 without changing predictions.

<<<<<<< SEARCH
        fusion_power = 1.174
=======
        fusion_power = 1.3515
>>>>>>> REPLACE