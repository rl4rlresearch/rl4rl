MECHANISM: Faster late-training exponential weight averaging

HYPOTHESIS: Raising the EMA interpolation coefficient to 0.02 will reproduce at least 9,259 correct predictions and reduce cross-entropy below the current 0.2135168.

INTENDED_EDIT: Double the EMA coefficient from 0.01 to 0.02 while preserving the architecture, schedule, exposure, and evaluation calibration.

EVIDENCE: The reported 0.02 EMA run achieved 9,259 correct predictions with 0.2126968 cross-entropy, improving on the current 0.01 design’s 9,258 correct and 0.2135168 cross-entropy.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.01)
=======
                    average.lerp_(parameter.detach(), 0.02)
>>>>>>> REPLACE