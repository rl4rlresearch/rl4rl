MECHANISM: Mild label-smoothing regularization

HYPOTHESIS: Applying 0.04 label smoothing will exceed 9,345 correct predictions by reducing overconfident fitting during the fixed two-example-per-training-image exposure without adding parameters or material runtime.

INTENDED_EDIT: Retain the successful architecture, batch size, optimizer, augmentation, and inference ensemble while changing the training objective to mildly smoothed cross-entropy.

EVIDENCE: Raw-mean gating reached the best 9,345 correct, while subsequent representation changes either regressed or exceeded the time limit; a computationally negligible loss-level intervention tests whether generalization rather than missing capacity is limiting accuracy.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels)
=======
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=0.04,
    )
>>>>>>> REPLACE