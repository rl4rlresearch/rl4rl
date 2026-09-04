MECHANISM: Late additive target-margin training

HYPOTHESIS: Ramping a 0.15 target-logit margin during EMA collection will exceed 9,328 correct predictions by maintaining pressure on borderline examples after label smoothing reaches zero.

INTENDED_EDIT: Leave the first half unchanged, then progressively subtract up to 0.15 from each target logit only inside the training loss.

EVIDENCE: The verified 9,328 design benefits from annealing label smoothing to zero in the second half, while added capacity and auxiliary supervision regressed; a small runtime-neutral margin directly tests whether stronger late decision-boundary optimization is more useful than further architectural complexity.

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
    target_margin = 0.15 * dropout_decay
    margin_logits = logits - target_margin * F.one_hot(
        labels,
        num_classes=10,
    ).to(dtype=logits.dtype)
    return F.cross_entropy(
        margin_logits,
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE