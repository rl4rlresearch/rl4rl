MECHANISM: Increased label-smoothing regularization

HYPOTHESIS: Raising label smoothing from 0.02 to 0.03 will increase validation_correct above 9,256 by improving generalization, opposite the harmful reduction to 0.01.

INTENDED_EDIT: Increase training label smoothing to 0.03 while preserving the best verified architecture, augmentation, ensemble weighting, and temperature.

EVIDENCE: Reducing label smoothing to 0.01 lowered validation_correct from 9,256 to 9,239, providing direct evidence that weaker smoothing is harmful and motivating a conservative test in the opposite direction.

<<<<<<< SEARCH
        label_smoothing=0.02,
=======
        label_smoothing=0.03,
>>>>>>> REPLACE