MECHANISM: Quadratic-extrapolated crop power-mean fusion

HYPOTHESIS: Raising the crop fusion power to 1.34 will preserve all 9,348 correct predictions while lowering validation cross-entropy below 0.18766820755004882.

INTENDED_EDIT: Move the crop-consensus power from 1.174 to the approximate minimum inferred from the measured arithmetic, 1.087, and 1.174 fusion results.

EVIDENCE: Cross-entropy improved monotonically from 0.18770656127929689 at power 1.0 to 0.1876840835571289 at 1.087 and 0.18766820755004882 at 1.174, with all three retaining 9,348 correct; their diminishing gains fit a local quadratic whose minimum is near 1.34.

<<<<<<< SEARCH
        fusion_power = 1.174
=======
        fusion_power = 1.34
>>>>>>> REPLACE