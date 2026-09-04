MECHANISM: Cosine-decayed label smoothing

HYPOTHESIS: Decaying label smoothing from 0.02 to zero will exceed 9,257 correct predictions by retaining early regularization while allowing late training to sharpen class boundaries for exact top-1 accuracy.

INTENDED_EDIT: Compute a progress-dependent smoothing coefficient and use it consistently in every individual, paired-view, and ensemble cross-entropy term.

EVIDENCE: The 233,434-parameter architecture remains best while architectural, augmentation, pooling, averaging, and batch-size changes regressed; the loss schedule is comparatively unexplored, and exact correct count is prioritized over calibration.

<<<<<<< SEARCH
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    offsets = torch.randint(0, 5, (2,))
=======
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    label_smoothing = 0.01 * (
        1.0 + math.cos(math.pi * progress)
    )
    offsets = torch.randint(0, 5, (2,))
>>>>>>> REPLACE

<<<<<<< SEARCH
    full_individual_loss = F.cross_entropy(
        torch.cat((logits_a, logits_b, flipped_a, flipped_b), dim=0),
        labels.repeat(4),
        label_smoothing=0.02,
    )
    central_individual_loss = F.cross_entropy(
        torch.cat((central_logits, flipped_central), dim=0),
        labels.repeat(2),
        label_smoothing=0.02,
    )
=======
    full_individual_loss = F.cross_entropy(
        torch.cat((logits_a, logits_b, flipped_a, flipped_b), dim=0),
        labels.repeat(4),
        label_smoothing=label_smoothing,
    )
    central_individual_loss = F.cross_entropy(
        torch.cat((central_logits, flipped_central), dim=0),
        labels.repeat(2),
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    full_pair_loss = F.cross_entropy(
        torch.cat(
            (
                0.5 * (logits_a + flipped_a),
                0.5 * (logits_b + flipped_b),
            ),
            dim=0,
        ),
        labels.repeat(2),
        label_smoothing=0.02,
    )
    central_pair_loss = F.cross_entropy(
        0.5 * (central_logits + flipped_central),
        labels,
        label_smoothing=0.02,
    )
=======
    full_pair_loss = F.cross_entropy(
        torch.cat(
            (
                0.5 * (logits_a + flipped_a),
                0.5 * (logits_b + flipped_b),
            ),
            dim=0,
        ),
        labels.repeat(2),
        label_smoothing=label_smoothing,
    )
    central_pair_loss = F.cross_entropy(
        0.5 * (central_logits + flipped_central),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    ensemble_loss = F.cross_entropy(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=0.02,
    )
=======
    ensemble_loss = F.cross_entropy(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE