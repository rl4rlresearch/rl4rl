MECHANISM: Zero-floor cosine convergence

HYPOTHESIS: Restoring static 0.02 label smoothing and lowering the cosine learning-rate floor from 2% to zero will exceed 9,330 correct predictions by letting the ensemble-dominant objective settle more fully late in training.

INTENDED_EDIT: Restore Reference Design 2’s winning linear loss curriculum and static smoothing, then change only its cosine-decay floor from 0.02 to 0.

EVIDENCE: Linear ensemble annealing with static 0.02 smoothing achieved 9,330 correct; terminal smoothing decay fell to 9,322, and raising the learning-rate floor from 2% to 5% fell further to 9,317, motivating restoration of static smoothing and testing the opposite floor direction.

<<<<<<< SEARCH
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
    label_smoothing = 0.02 * min(1.0, 4.0 * (1.0 - progress))
=======
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
    label_smoothing = 0.02
>>>>>>> REPLACE

<<<<<<< SEARCH
        multiplier = 0.02 + 0.98 * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
=======
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE