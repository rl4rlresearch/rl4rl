MECHANISM: Mild label-smoothed supervision

HYPOTHESIS: Applying 0.05 label smoothing will improve generalization enough to achieve at least 9,311 correct validation predictions while preserving the verified architecture and runtime profile.

INTENDED_EDIT: Replace hard-label cross-entropy with mildly label-smoothed cross-entropy; leave the 249,601-parameter model, optimizer, schedule, averaging, augmentation, and TTA unchanged.

EVIDENCE: The normalized 147-unit head reached the current best 9,310 correct, while architectural changes regressed or timed out and reducing tail averaging fell to 9,307; this motivates an orthogonal, negligible-cost training-loss change that preserves every verified inference component.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
>>>>>>> REPLACE