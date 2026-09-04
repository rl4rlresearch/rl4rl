MECHANISM: Reduced tail-checkpoint interpolation

HYPOTHESIS: Reducing tail-average weight from 0.5 to 0.25 will preserve the final iterate more strongly and achieve at least 9,311 validation-correct predictions.

INTENDED_EDIT: Change only the final parameter interpolation weight, retaining the verified architecture, sequential TTA, optimizer, and runtime profile.

EVIDENCE: Raising tail-average weight to 0.75 reduced validation-correct from 9,286 to 9,276. Prior 0.25 tests were confounded by slower batched TTA and timed out, while the current sequential design completed with 9,310 correct.

<<<<<<< SEARCH
                    parameter.lerp_(average, 0.5)
=======
                    parameter.lerp_(average, 0.25)
>>>>>>> REPLACE