MECHANISM: Slightly stronger fixed label smoothing

HYPOTHESIS: Raising fixed label smoothing from 0.02 to 0.03 will exceed 9,330 correct predictions by strengthening the regularization that was lost when smoothing was annealed toward zero.

INTENDED_EDIT: Preserve the verified architecture, optimizer, curriculum, schedule, and evaluation ensemble while increasing label smoothing for both ensemble and per-view losses.

EVIDENCE: Annealing smoothing from 0.02 to zero reduced validation correctness from 9,330 to 9,312 despite lower cross-entropy, indicating that maintaining soft targets benefits the primary accuracy objective.

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
        label_smoothing=0.03,
    )
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.03)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.03)
    )
>>>>>>> REPLACE