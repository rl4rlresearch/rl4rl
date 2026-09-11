MECHANISM: Low-dose annealed label smoothing

HYPOTHESIS: Halving the initial smoothing strength to 0.025 will exceed 9,345 correct predictions by retaining early output-space regularization while reducing the optimization bias that raised cross-entropy at 0.05.

INTENDED_EDIT: Linearly decay label smoothing from 0.025 to zero across the fixed training schedule.

EVIDENCE: Annealed smoothing at 0.05 reached 9,341 correct—only four below the 9,345 baseline—while worsening cross-entropy, indicating that the mechanism was nearly accuracy-neutral but likely too strong.

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels)
=======
    progress = min(step / max(total_steps - 1, 1), 1.0)
    label_smoothing = 0.025 * (1.0 - progress)
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE