MECHANISM: Cosine target-confidence curriculum

HYPOTHESIS: Annealing label smoothing from 0.04 to zero will exceed 9,167 correct predictions by regularizing early broad-translation learning while sharpening class boundaries during terminal inference-aligned training.

INTENDED_EDIT: Preserve the verified 39/64 curriculum and all other settings, but replace fixed 0.02 label smoothing with a cosine schedule having the same training-wide mean.

EVIDENCE: The 39/64 design achieved the best completed result at 9,167 correct, and progressively longer inference-aligned terminal phases consistently improved accuracy; this motivates concentrating soft-target regularization early and using hard targets late.

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
) -> torch.Tensor:
    progress = min(step / max(total_steps, 1), 1.0)
    label_smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE