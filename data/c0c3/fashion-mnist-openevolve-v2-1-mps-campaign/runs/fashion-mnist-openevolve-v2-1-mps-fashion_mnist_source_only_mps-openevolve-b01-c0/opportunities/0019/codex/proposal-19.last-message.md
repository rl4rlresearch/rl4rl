MECHANISM: Reduced target smoothing

HYPOTHESIS: Halving label smoothing from 0.02 to 0.01 will exceed 9,290 correct predictions by preserving mild regularization while strengthening true-class gradients during the limited two-pass exposure.

INTENDED_EDIT: Reduce cross-entropy label smoothing from 2% to 1%, leaving the verified architecture, dropout, augmentation, optimizer, EMA, and evaluation ensemble unchanged.

EVIDENCE: The 58-unit dense head reached 9,290 correct, while removing dropout fell to 9,270 and increasing it to 15% fell to 9,266; this supports retaining the verified feature regularization and making a controlled reduction to the separate target-smoothing regularizer.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.01)
>>>>>>> REPLACE