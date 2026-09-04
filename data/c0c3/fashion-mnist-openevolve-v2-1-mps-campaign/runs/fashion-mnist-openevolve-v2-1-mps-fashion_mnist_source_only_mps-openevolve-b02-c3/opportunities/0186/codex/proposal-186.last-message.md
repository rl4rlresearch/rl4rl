MECHANISM: Half-strength super-arithmetic crop fusion

HYPOTHESIS: Setting the crop fusion power to 1.087 will exceed 9,348 correct predictions if beneficial boundary crossings between the equally accurate arithmetic and 1.174-order endpoints occur before offsetting regressions.

INTENDED_EDIT: Interpolate the crop-level power mean halfway from arithmetic fusion toward the current 1.174-order fusion while preserving orientation averaging, crop weights, training, and calibration.

EVIDENCE: Reference Design 1’s arithmetic fusion and the current 1.174-order crop fusion both achieved 9,348 correct, while the latter reduced cross-entropy; unlike the unsuccessful orientation-power midpoint, this tests the still-unprobed crop-consensus axis.

<<<<<<< SEARCH
        fusion_power = 1.174
=======
        fusion_power = 1.087
>>>>>>> REPLACE