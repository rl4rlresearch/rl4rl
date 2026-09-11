MECHANISM: Hard-label objective for short-horizon convergence

HYPOTHESIS: Removing 0.02 label smoothing will exceed 9,290 correct predictions by strengthening gradients toward the true class during the fixed two-pass training horizon; if correctness ties, sharper probabilities should also reduce validation cross-entropy.

INTENDED_EDIT: Retain the strongest verified architecture, batch size, optimizer, and schedule while switching training loss to ordinary cross-entropy.

EVIDENCE: Batch size 64 produced the best result at 9,290 correct, while architecture expansion, augmentation, EMA, test-time reflection, and finer batches all regressed or timed out; the remaining label smoothing is an untested regularizer that may impede convergence under limited exposure.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    return F.cross_entropy(model(images), labels)
>>>>>>> REPLACE