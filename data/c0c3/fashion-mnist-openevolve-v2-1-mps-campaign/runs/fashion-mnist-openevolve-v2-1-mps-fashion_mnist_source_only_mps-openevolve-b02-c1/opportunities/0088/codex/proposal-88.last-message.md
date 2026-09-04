MECHANISM: Stronger late-training weight averaging

HYPOTHESIS: Increasing the tail-average interpolation from 0.5 to 0.75 will exceed 9,286 correct predictions by reducing final-update variance without increasing parameters or computation.

INTENDED_EDIT: Give the arithmetic average of late-training checkpoints 75% rather than 50% weight in the final model.

EVIDENCE: Symmetric TTA reweighting plateaued at 9,286 correct or regressed to 9,281, while representation changes repeatedly timed out; adjusting the existing tail average changes the learned decision boundaries at negligible runtime cost.

<<<<<<< SEARCH
                    parameter.lerp_(average, 0.5)
=======
                    parameter.lerp_(average, 0.75)
>>>>>>> REPLACE