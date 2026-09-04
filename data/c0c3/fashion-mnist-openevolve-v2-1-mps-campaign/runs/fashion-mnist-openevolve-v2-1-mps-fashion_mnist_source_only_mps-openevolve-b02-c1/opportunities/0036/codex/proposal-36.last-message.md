MECHANISM: Reduced-strength tail interpolation

HYPOTHESIS: Blending 40% toward the sparse tail mean will exceed 9,286 correct predictions by retaining most of the variance-reduction benefit while preserving more of the final iterate’s decision boundaries.

INTENDED_EDIT: Reduce the final tail-average interpolation coefficient from 0.5 to 0.4.

EVIDENCE: Half-strength interpolation improved validation_correct from the established 9,280 to 9,286, while full tail averaging previously lost a prediction; this motivates refining the averaging strength toward the final iterate.

<<<<<<< SEARCH
                    parameter.lerp_(average, 0.5)
=======
                    parameter.lerp_(average, 0.4)
>>>>>>> REPLACE