MECHANISM: Mild uniform label smoothing

HYPOTHESIS: Applying 0.03 label smoothing will exceed 9,360 correct predictions by regularizing overconfident class boundaries while preserving the verified architecture, EMA, augmentation, TTA, and calibration.

INTENDED_EDIT: Replace hard-label cross-entropy with mildly smoothed cross-entropy.

EVIDENCE: Confidence-based focal reweighting reduced validation_correct to 9,331, while architecture, EMA, and TTA changes also underperformed; this motivates retaining uniform example weighting and testing a gentler target-level regularizer.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels)
=======
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=0.03,
    )
>>>>>>> REPLACE