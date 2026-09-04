MECHANISM: Annealed label smoothing

HYPOTHESIS: Linearly reducing label smoothing from 0.02 to zero will exceed 9,330 correct predictions by retaining early regularization while aligning late updates with the hard-label accuracy objective, without increasing runtime.

INTENDED_EDIT: Reuse training progress to anneal label smoothing in both ensemble and per-view cross-entropy losses.

EVIDENCE: The current fixed-smoothing design verified 9,330 correct in 77.44 seconds, while added augmentation, consistency, architecture, and EMA work repeatedly timed out; this isolates a zero-overhead loss-curriculum change.

<<<<<<< SEARCH
    ensemble_weight = 0.5 + 0.5 * step / max(total_steps - 1, 1)
=======
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
    label_smoothing = 0.02 * (1.0 - progress)
>>>>>>> REPLACE

<<<<<<< SEARCH
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
=======
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.02)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.02)
    )
=======
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