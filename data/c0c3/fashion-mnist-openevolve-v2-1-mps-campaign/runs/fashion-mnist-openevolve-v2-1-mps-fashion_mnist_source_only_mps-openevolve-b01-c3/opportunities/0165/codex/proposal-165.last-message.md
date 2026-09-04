MECHANISM: Pair-ensemble aligned supervision

HYPOTHESIS: Blending 25% classification loss on the averaged paired logits will exceed 9,312 correct predictions by directly optimizing the same logit-averaging behavior used during validation.

INTENDED_EDIT: Replace separate per-view classification losses with an equivalent fused individual-view loss blended with cross-entropy on the pair’s mean logits; retain augmentation, consistency regularization, inference pooling, and runtime profile.

EVIDENCE: Stochastic flip-offset pairing retained 9,312 correct while reducing cross-entropy to 0.193848, whereas learned feature fusion fell to 9,280; this motivates supervising the successful output-level pair average without changing the architecture.

<<<<<<< SEARCH
    classification_loss = 0.5 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=0.02,
        )
        + F.cross_entropy(
            partner_logits,
            labels,
            label_smoothing=0.02,
        )
    )
=======
    individual_classification_loss = F.cross_entropy(
        paired_logits,
        labels.repeat(2),
        label_smoothing=0.02,
    )
    ensemble_classification_loss = F.cross_entropy(
        0.5 * (original_logits + partner_logits),
        labels,
        label_smoothing=0.02,
    )
    classification_loss = (
        0.75 * individual_classification_loss
        + 0.25 * ensemble_classification_loss
    )
>>>>>>> REPLACE