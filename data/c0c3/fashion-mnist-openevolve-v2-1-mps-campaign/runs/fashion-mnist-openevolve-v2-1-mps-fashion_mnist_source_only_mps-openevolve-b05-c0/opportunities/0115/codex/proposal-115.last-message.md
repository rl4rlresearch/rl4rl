MECHANISM: Faster late-training exponential weight averaging

HYPOTHESIS: Increasing the EMA coefficient from 0.01 to 0.02 will reproduce at least 9,259 correct predictions and lower validation cross-entropy versus the current 0.01 design.

INTENDED_EDIT: Double the EMA interpolation coefficient while leaving architecture, training exposure, and evaluation calibration unchanged.

EVIDENCE: The reported 0.02 EMA run achieved 9,259 correct predictions and 0.2126968 cross-entropy, improving on the current design’s 9,258 correct and 0.2135168 cross-entropy.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.01)
=======
                    average.lerp_(parameter.detach(), 0.02)
>>>>>>> REPLACE