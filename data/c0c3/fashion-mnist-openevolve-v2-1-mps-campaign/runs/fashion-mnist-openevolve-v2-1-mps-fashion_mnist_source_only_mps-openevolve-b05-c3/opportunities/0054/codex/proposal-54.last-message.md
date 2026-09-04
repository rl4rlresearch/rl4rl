MECHANISM: Late hard-label margin refinement

HYPOTHESIS: Retaining the best 19/32 augmentation curriculum while disabling label smoothing only for the final eighth of training will exceed 9,172 correct predictions by sharpening decision boundaries at low learning rates without giving up early regularization.

INTENDED_EDIT: Restore the verified 19/32 broad-to-cardinal transition and use hard-label cross-entropy during the final eighth of training.

EVIDENCE: The 19/32 curriculum achieved the best completed result at 9,172 correct, while head widening and AdamW β₂ changes were weaker; the existing augmentation, dropout, weight decay, and EMA permit a focused late loss refinement without architectural cost.

<<<<<<< SEARCH
    if step * 64 < total_steps * 39:
=======
    if step * 32 < total_steps * 19:
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
) -> torch.Tensor:
    label_smoothing = 0.0 if step * 8 >= total_steps * 7 else 0.02
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE