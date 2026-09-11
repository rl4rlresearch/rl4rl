MECHANISM: Lower-side endpoint–EMA interpolation refinement

HYPOTHESIS: Blending 18.75% of the bias-corrected 0.99 EMA into the endpoint will exceed 9,130 correct predictions by retaining more of the endpoint than the successful 25% blend while preserving useful trajectory smoothing.

INTENDED_EDIT: Reduce the final EMA interpolation coefficient from 0.25 to 0.1875 without changing training, architecture, or evaluation.

EVIDENCE: The 25% blend improved accuracy from 9,122 to 9,130, whereas 50% and 100% blends fell to 9,123 and 9,121; this brackets the best observed region at or below 25% and motivates a lower-side refinement.

<<<<<<< SEARCH
                parameter.lerp_(shadow / correction, 0.25)
=======
                parameter.lerp_(shadow / correction, 0.1875)
>>>>>>> REPLACE