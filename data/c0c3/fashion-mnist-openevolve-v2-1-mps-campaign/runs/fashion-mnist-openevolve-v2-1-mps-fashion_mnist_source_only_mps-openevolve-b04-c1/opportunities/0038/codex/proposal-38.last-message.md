MECHANISM: Per-view auxiliary supervision for mirrored ensemble

HYPOTHESIS: Mixing 10% individual-view cross-entropy into the mirrored-average objective will exceed 9,286 correct predictions by making both constituent views independently discriminative while retaining direct optimization of their ensemble.

INTENDED_EDIT: Replace the ensemble-only loss with a scale-preserving blend of 90% mirrored-average loss and 10% mean per-view loss; architecture, augmentation, EMA, schedule, and evaluation calibration remain unchanged.

EVIDENCE: Mirrored-view ensembling previously improved validation correct from 9,237 to 9,282, while temperature calibration has now saturated at 9,286; auxiliary supervision directly tests whether stronger constituent predictions can improve the proven ensemble without additional forward passes or parameters.

<<<<<<< SEARCH
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
=======
    ensemble_loss = F.cross_entropy(
        logits,
        labels,
        label_smoothing=0.02,
    )
    view_loss = 0.5 * (
        F.cross_entropy(
            paired_logits[:batch_size],
            labels,
            label_smoothing=0.02,
        )
        + F.cross_entropy(
            paired_logits[batch_size:],
            labels,
            label_smoothing=0.02,
        )
    )
    return 0.90 * ensemble_loss + 0.10 * view_loss
>>>>>>> REPLACE