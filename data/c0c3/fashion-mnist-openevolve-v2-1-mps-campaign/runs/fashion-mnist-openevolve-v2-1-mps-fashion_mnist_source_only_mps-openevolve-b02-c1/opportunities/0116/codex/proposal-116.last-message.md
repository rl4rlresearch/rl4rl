MECHANISM: Stronger tail-checkpoint interpolation

HYPOTHESIS: Increasing the tail-average blend from 0.5 to 0.75 will achieve at least 9,312 correct predictions by moving farther toward the better-generalizing averaged parameters.

INTENDED_EDIT: Strengthen only the final interpolation toward the existing tail parameter average, without changing computation, parameters, BatchNorm buffers, or TTA.

EVIDENCE: A 0.5 tail-average blend produced 9,311 correct, while reducing it to 0.25 regressed to 9,307; this directional result motivates testing a stronger blend.

<<<<<<< SEARCH
                    parameter.lerp_(average, 0.5)
=======
                    parameter.lerp_(average, 0.75)
>>>>>>> REPLACE