MECHANISM: Fine-grained endpoint–EMA interpolation

HYPOTHESIS: Blending 21.875% of the bias-corrected EMA into the endpoint will exceed 9,130 correct predictions by refining the narrow optimum bracketed by the 18.75% and 25% results.

INTENDED_EDIT: Reduce the final EMA interpolation coefficient from 0.25 to 0.21875 while leaving training and evaluation unchanged.

EVIDENCE: The 25% blend achieved 9,130 correct, while 18.75% achieved 9,129 and 50% fell to 9,123, indicating a narrow optimum near 25% and motivating a midpoint refinement on its better-supported lower side.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.25)
=======
                parameter.lerp_(shadow / correction, 0.21875)
>>>>>>> REPLACE