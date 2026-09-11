MECHANISM: Reduced-strength sparse tail interpolation

HYPOTHESIS: Blending 40% toward the sparse tail mean will exceed 9,286 correct predictions by preserving more final-iterate decision boundaries while retaining late-training variance reduction.

INTENDED_EDIT: Reduce the final tail-average interpolation coefficient from 0.5 to 0.4.

EVIDENCE: Half-strength interpolation improved validation_correct from 9,280 to 9,286, while full averaging lost a prediction; prior 0.4 attempts timed out and therefore did not test its accuracy effect.

<<<<<<< SEARCH
                    parameter.lerp_(average, 0.5)
=======
                    parameter.lerp_(average, 0.4)
>>>>>>> REPLACE