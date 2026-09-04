MECHANISM: Conservative increase in label-smoothing regularization

HYPOTHESIS: Raising label smoothing from 0.02 to 0.025 will improve generalization enough to exceed 9,256 correct predictions while avoiding the larger unverified jump to 0.03.

INTENDED_EDIT: Increase only the training loss label-smoothing coefficient, preserving the best verified architecture, optimizer, augmentation, ensemble, and calibration.

EVIDENCE: Reducing label smoothing to 0.01 lowered validation_correct from 9,256 to 9,239, directly indicating that weaker smoothing is harmful; 0.025 tests the favorable direction conservatively.

<<<<<<< SEARCH
        label_smoothing=0.02,
=======
        label_smoothing=0.025,
>>>>>>> REPLACE