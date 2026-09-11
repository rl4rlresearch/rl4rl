MECHANISM: Lower-boundary EMA interpolation refinement

HYPOTHESIS: Blending 27.34375% of the bias-corrected EMA into the endpoint will exceed 9,133 correct predictions by testing symmetrically below the best 28.125% blend after the equally distant 28.90625% blend lost one correct prediction.

INTENDED_EDIT: Reduce the final EMA interpolation coefficient from 0.28125 to 0.2734375 without changing architecture, training, or evaluation.

EVIDENCE: The 28.125% blend achieved the best result with 9,133 correct, while 28.90625% achieved 9,132; testing the symmetric lower-side coefficient is the most direct refinement of the apparent local optimum.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.28125)
=======
                parameter.lerp_(shadow / correction, 0.2734375)
>>>>>>> REPLACE