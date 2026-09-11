MECHANISM: Late additive-margin handoff

HYPOTHESIS: Ramping a 0.05 target-logit margin as label smoothing decays will exceed 9,328 correct predictions by preserving beneficial early regularization while improving late class-boundary separation during EMA collection.

INTENDED_EDIT: Keep the architecture and training schedule unchanged, but linearly introduce an additive classification margin over the second half of training.

EVIDENCE: Ending smoothing earlier fell to 9,316 correct, while retaining more smoothing late reached only 9,325; this motivates preserving the verified smoothing schedule while strengthening the late hard-label objective with negligible runtime cost.

<<<<<<< SEARCH
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    return F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )
=======
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    classification_margin = 0.05 * dropout_decay
    margin_targets = F.one_hot(
        labels,
        num_classes=logits.shape[1],
    ).to(dtype=logits.dtype)
    margin_logits = logits - classification_margin * margin_targets
    return F.cross_entropy(
        margin_logits,
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE