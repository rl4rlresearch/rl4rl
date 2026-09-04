MECHANISM: Annealed target-distribution smoothing

HYPOTHESIS: Linearly decaying label smoothing from 0.05 to zero will exceed 9,345 correct predictions by regularizing early decision-boundary formation while restoring exact hard-label optimization near convergence.

INTENDED_EDIT: Replace hard-label cross-entropy with label smoothing that decreases continuously over the fixed training schedule.

EVIDENCE: Dropout 0.10 achieved 9,345 correct, while both reducing it to 0.05 and increasing it to 0.15 or 0.1125 regressed, motivating an orthogonal output-space regularizer whose influence vanishes late rather than another persistent dropout change.

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels)
=======
) -> torch.Tensor:
    progress = min(step / max(total_steps - 1, 1), 1.0)
    label_smoothing = 0.05 * (1.0 - progress)
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE