MECHANISM: Faster late-training exponential weight averaging

HYPOTHESIS: Increasing the EMA coefficient to 0.02 will reproduce at least 9,259 correct predictions and reduce validation cross-entropy below 0.2135168.

INTENDED_EDIT: Double the EMA interpolation coefficient while preserving the architecture, schedule, exposure, and evaluation calibration.

EVIDENCE: A completed 0.02 EMA run achieved 9,259 correct predictions and 0.2126968 cross-entropy, improving on the current 0.01 design’s 9,258 correct and 0.2135168 cross-entropy.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.01)
=======
                    average.lerp_(parameter.detach(), 0.02)
>>>>>>> REPLACE