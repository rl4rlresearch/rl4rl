MECHANISM: Slightly stronger uniform target regularization

HYPOTHESIS: Raising constant label smoothing from 0.02 to 0.025 will exceed 9,331 correct predictions by improving generalization, while restoring the best verified flip-fusion coefficients.

INTENDED_EDIT: Restore the independently verified inference fusion weight and slightly increase label smoothing for both ensemble and individual-view losses.

EVIDENCE: Annealing smoothing toward zero reduced correctness from 9,331 to 9,321 while lowering cross-entropy, indicating that weaker target regularization trades away the primary ranking metric; a modest increase tests the favorable direction without changing runtime or architecture.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.4914921875 * logits + 0.5085078125 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.4914990234375 * logits + 0.5085009765625 * flipped_logits
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress

    paired_images = torch.cat(
=======
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
    label_smoothing = 0.025

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