MECHANISM: Fractional center-weight translation ensembling at the verified EMA horizon

HYPOTHESIS: A centered-crop weight of 2.75 with decay 0.98 will retain or exceed 9,344 correct predictions while lowering cross-entropy below the weight-3 baseline.

INTENDED_EDIT: Restore the verified 0.98 full-state EMA and interpolate the centered TTA crop weight halfway between 2.5 and 3.0.

EVIDENCE: Weight 3 with decay 0.98 achieved the best observed 9,344 correct, while weight 2.5 reduced cross-entropy to 0.18778313 but lost one correct prediction; 2.75 directly tests the remaining boundary between accuracy and calibration.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (2.75, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
                decay = 0.975 ** elapsed
=======
                decay = 0.98 ** elapsed
>>>>>>> REPLACE