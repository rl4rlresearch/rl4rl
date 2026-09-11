MECHANISM: Reduced-strength sparse tail interpolation

HYPOTHESIS: Blending 40% toward the sparse tail mean will exceed 9,286 correct predictions by preserving more of the final iterate’s decision boundaries while retaining useful late-training variance reduction.

INTENDED_EDIT: Reduce the final tail-average interpolation coefficient from 0.5 to 0.4.

EVIDENCE: Half-strength interpolation improved validation_correct from 9,280 to 9,286, while full tail averaging previously lost a prediction; the prior 0.4 verification timed out and therefore did not test this accuracy hypothesis.

<<<<<<< SEARCH
                    parameter.lerp_(average, 0.5)
=======
                    parameter.lerp_(average, 0.4)
>>>>>>> REPLACE