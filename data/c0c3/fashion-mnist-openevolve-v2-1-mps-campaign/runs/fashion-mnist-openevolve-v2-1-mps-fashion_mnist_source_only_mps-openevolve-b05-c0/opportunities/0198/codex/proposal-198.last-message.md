MECHANISM: Reduced target smoothing for faster discriminative fitting

HYPOTHESIS: Lowering label smoothing from 0.05 to 0.04 will increase validation_correct above 9,260 by strengthening the correct-class training signal within the fixed two-epoch exposure budget.

INTENDED_EDIT: Change only the cross-entropy label-smoothing coefficient, preserving model size, runtime profile, optimizer, EMA, and evaluation calibration.

EVIDENCE: Compute-increasing architecture and test-time ensemble changes repeatedly timed out, while translation augmentation reduced accuracy to 9,215; a compute-neutral loss adjustment is the most informative remaining way to improve the verified training trajectory.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
>>>>>>> REPLACE