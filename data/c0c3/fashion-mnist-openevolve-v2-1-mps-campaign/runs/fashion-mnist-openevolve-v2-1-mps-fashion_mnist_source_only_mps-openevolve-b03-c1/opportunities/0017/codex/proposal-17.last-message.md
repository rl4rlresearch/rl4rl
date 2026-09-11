MECHANISM: Upper-side endpoint–EMA interpolation refinement

HYPOTHESIS: Blending 28.125% of the bias-corrected 0.99 EMA into the endpoint will exceed 9,130 correct predictions by moving slightly beyond the best observed 25% blend while remaining well below the degraded 50% blend.

INTENDED_EDIT: Increase the final EMA interpolation coefficient from 0.25 to 0.28125 without changing training, architecture, or evaluation.

EVIDENCE: Accuracy increased from 9,129 at 18.75% blending to 9,130 at both 21.875% and 25%, with 25% also giving better cross-entropy; 50% fell to 9,123, so 28.125% is the nearest untested upper-side refinement around the apparent optimum.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.25)
=======
                parameter.lerp_(shadow / correction, 0.28125)
>>>>>>> REPLACE