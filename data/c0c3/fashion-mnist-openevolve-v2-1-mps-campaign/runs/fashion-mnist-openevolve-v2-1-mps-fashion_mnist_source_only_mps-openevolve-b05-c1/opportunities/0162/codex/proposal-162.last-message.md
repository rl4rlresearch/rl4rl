MECHANISM: Mild uniform label smoothing

HYPOTHESIS: Applying 0.03 label smoothing will exceed 9,360 correct predictions by gently regularizing class boundaries while preserving the verified architecture, EMA, augmentation, and logit-space TTA.

INTENDED_EDIT: Replace hard-target cross-entropy with cross-entropy using 0.03 label smoothing.

EVIDENCE: The only prior verification of this exact low-risk loss change timed out, so it provides no performance evidence against the idea; larger changes to architecture, EMA, focal weighting, and probability-space TTA all reduced validation_correct.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels)
=======
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=0.03,
    )
>>>>>>> REPLACE