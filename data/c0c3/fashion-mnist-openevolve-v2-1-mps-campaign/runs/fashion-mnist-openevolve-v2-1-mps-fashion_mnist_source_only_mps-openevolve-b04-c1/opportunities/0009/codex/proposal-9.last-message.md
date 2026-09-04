MECHANISM: Auxiliary per-view supervision for paired ensemble training

HYPOTHESIS: Blending 20% individual-view cross-entropy into the proven paired-logit objective will exceed 9,282 correct predictions by discouraging view-specific errors while retaining direct optimization of the validation ensemble.

INTENDED_EDIT: Compute cross-entropy for both original and mirrored logits and blend it with the existing ensemble loss at an 80/20 ratio.

EVIDENCE: Paired-view ensemble training improved validation correct from 9,237 to 9,282; auxiliary supervision is a focused test of whether ensuring both constituent views remain discriminative can further strengthen that successful predictor.

<<<<<<< SEARCH
    logits = 0.5 * (
        paired_logits[:batch_size] + paired_logits[batch_size:]
    )
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
=======
    logits = 0.5 * (
        paired_logits[:batch_size] + paired_logits[batch_size:]
    )
    ensemble_loss = F.cross_entropy(
        logits,
        labels,
        label_smoothing=0.02,
    )
    view_loss = F.cross_entropy(
        paired_logits,
        labels.repeat(2),
        label_smoothing=0.02,
    )
    return 0.80 * ensemble_loss + 0.20 * view_loss
>>>>>>> REPLACE