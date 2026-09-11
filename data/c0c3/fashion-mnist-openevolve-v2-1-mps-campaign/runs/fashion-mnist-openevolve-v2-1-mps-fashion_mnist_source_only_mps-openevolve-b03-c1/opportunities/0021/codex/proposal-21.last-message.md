MECHANISM: Upper-boundary EMA interpolation refinement

HYPOTHESIS: Blending 28.90625% of the bias-corrected EMA into the endpoint will retain 9,133 correct predictions while lowering cross-entropy below 0.2466727, exceeding the current validation score.

INTENDED_EDIT: Increase the final EMA interpolation coefficient from 0.28125 to 0.2890625 without changing training, architecture, or evaluation.

EVIDENCE: The 28.125% blend achieved 9,133 correct, while 29.6875% lost only one correct prediction but improved cross-entropy; their midpoint tests whether more smoothing can improve the tie-breaker before accuracy declines.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.28125)
=======
                parameter.lerp_(shadow / correction, 0.2890625)
>>>>>>> REPLACE