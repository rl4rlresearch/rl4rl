MECHANISM: Upper-side endpoint–EMA interpolation refinement

HYPOTHESIS: Blending 31.25% of the bias-corrected 0.99 EMA into the endpoint will exceed 9,133 correct predictions by continuing the improvement from 25% to 28.125% while remaining substantially below the degraded 50% blend.

INTENDED_EDIT: Increase the final EMA interpolation coefficient from 0.28125 to 0.3125 without changing architecture, training, or evaluation.

EVIDENCE: Accuracy rose from 9,130 at 25% EMA blending to 9,133 at 28.125%, while 50% blending fell to 9,123; the nearest untested upper-side coefficient is therefore the most informative refinement of the apparent interior optimum.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.28125)
=======
                parameter.lerp_(shadow / correction, 0.3125)
>>>>>>> REPLACE