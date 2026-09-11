MECHANISM: Reduced-strength sparse tail interpolation

HYPOTHESIS: Blending 40% toward the sparse tail mean will exceed 9,286 correct predictions by preserving more of the final iterate’s decision boundaries while retaining useful late-training variance reduction.

INTENDED_EDIT: Reduce the final tail-average interpolation coefficient from 0.5 to 0.4.

EVIDENCE: Half-strength interpolation improved validation_correct from 9,280 to 9,286, while full averaging previously lost a prediction; both prior 0.4 verifications timed out and therefore left its accuracy effect unresolved.

<<<<<<< SEARCH
                for parameter, average in zip(
                    parameters, optimizer._tail_average
                ):
                    parameter.lerp_(average, 0.5)
=======
                for parameter, average in zip(
                    parameters, optimizer._tail_average
                ):
                    parameter.lerp_(average, 0.4)
>>>>>>> REPLACE