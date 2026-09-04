MECHANISM: Local endpoint–EMA interpolation refinement

HYPOTHESIS: Blending 29.6875% of the bias-corrected 0.99 EMA into the endpoint will exceed 9,133 correct predictions by refining the interval between the best 28.125% blend and the slightly degraded 31.25% blend.

INTENDED_EDIT: Increase the final EMA interpolation coefficient from 0.28125 to 0.296875 without changing architecture, training, or evaluation.

EVIDENCE: The 28.125% blend achieved 9,133 correct, while 31.25% achieved 9,132 and 25% achieved 9,130; the midpoint of the narrow best-supported interval is the most informative next coefficient.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.28125)
=======
                parameter.lerp_(shadow / correction, 0.296875)
>>>>>>> REPLACE