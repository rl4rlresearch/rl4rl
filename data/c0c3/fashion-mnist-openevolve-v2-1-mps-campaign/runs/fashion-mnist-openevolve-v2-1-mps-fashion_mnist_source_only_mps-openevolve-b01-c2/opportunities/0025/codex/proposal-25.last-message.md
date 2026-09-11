MECHANISM: Back-loaded label-smoothing curriculum

HYPOTHESIS: Increasing label smoothing linearly from 0.01 to 0.03 while preserving its 0.02 average will exceed 9,330 correct predictions by concentrating regularization late, when supervision becomes purely ensemble-aligned.

INTENDED_EDIT: Retain the winning architecture, optimizer, and linear ensemble-loss curriculum, but replace static 0.02 label smoothing with a same-average linear ramp.

EVIDENCE: Static 0.02 smoothing with linear ensemble annealing achieved 9,330 correct, whereas decaying smoothing during the final quarter fell to 9,322; this suggests late smoothing is valuable and motivates testing the opposite temporal allocation without changing average strength.

<<<<<<< SEARCH
) -> torch.Tensor:
    ensemble_weight = 0.5 + 0.5 * step / max(total_steps - 1, 1)
    paired_images = torch.cat(
=======
) -> torch.Tensor:
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
    label_smoothing = 0.01 + 0.02 * progress
    paired_images = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.02)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.02)
    )
=======
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=label_smoothing,
    )
    view_loss = 0.5 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=label_smoothing,
        )
        + F.cross_entropy(
            flipped_logits,
            labels,
            label_smoothing=label_smoothing,
        )
    )
>>>>>>> REPLACE