MECHANISM: Reduced late-checkpoint averaging

HYPOTHESIS: Reducing tail-average interpolation to 0.25 will retain variance reduction while reversing the boundary drift seen at 0.75, yielding at least 9,287 correct predictions.

INTENDED_EDIT: Give the late-training checkpoint average 25% rather than 50% weight in the final model.

EVIDENCE: Increasing interpolation from 0.5 to 0.75 lowered validation-correct from 9,286 to 9,276 despite improving cross-entropy, indicating that movement toward the tail average harms the primary accuracy objective and motivating a step in the opposite direction.

<<<<<<< SEARCH
                    parameter.lerp_(average, 0.5)
=======
                    parameter.lerp_(average, 0.25)
>>>>>>> REPLACE