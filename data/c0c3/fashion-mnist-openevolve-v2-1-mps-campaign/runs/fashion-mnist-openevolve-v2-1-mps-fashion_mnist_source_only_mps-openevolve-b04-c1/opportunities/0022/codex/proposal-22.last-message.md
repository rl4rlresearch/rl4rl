MECHANISM: Hard-target paired-view supervision

HYPOTHESIS: Removing label smoothing will exceed 9,286 correct predictions by accelerating class-boundary learning during the fixed two training exposures while EMA and dropout retain regularization.

INTENDED_EDIT: Change the mirrored-ensemble loss from 2% label-smoothed cross-entropy to ordinary hard-label cross-entropy.

EVIDENCE: Probability-space view averaging lowered validation cross-entropy but reduced correct predictions from 9,286 to 9,280, showing that softer calibration does not improve the primary objective; hard targets directly test whether sharper discrimination raises accuracy.

<<<<<<< SEARCH
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
=======
    return F.cross_entropy(logits, labels)
>>>>>>> REPLACE