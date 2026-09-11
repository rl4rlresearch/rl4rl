MECHANISM: Conservative centered-view interpolation

HYPOTHESIS: Raising the centered-view weight to 3.03125 will preserve 9,210 correct predictions while lowering validation cross-entropy, producing a strict validation-score improvement.

INTENDED_EDIT: Move one quarter of the way from the current 3.0 centered-view weight toward the tested 3.125 weight, retaining arithmetic probability pooling and all training behavior.

EVIDENCE: Weight 3.125 reduced cross-entropy from 0.23116648 to 0.23110609 but lost one correct prediction; a smaller interpolation tests whether its calibration benefit can be captured before crossing that decision boundary.

<<<<<<< SEARCH
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.03125, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE