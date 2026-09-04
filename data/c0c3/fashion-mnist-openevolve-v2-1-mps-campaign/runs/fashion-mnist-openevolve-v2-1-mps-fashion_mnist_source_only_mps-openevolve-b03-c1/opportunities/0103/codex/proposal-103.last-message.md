MECHANISM: Translation-favoring logit-space ensemble

HYPOTHESIS: Reducing the centered-view weight from 3.0 to 2.5 will exceed 9,166 correct predictions by modestly strengthening the contribution of the successful cardinal-shift ensemble.

INTENDED_EDIT: Decrease only the centered validation view’s weight while preserving training, cardinal views, horizontal flips, calibration, and parameter count.

EVIDENCE: Increasing the centered-view weight from 3.0 to 4.0 reduced correctness from 9,166 to 9,164, directly motivating a conservative test in the opposite direction.

<<<<<<< SEARCH
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (2.5, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE