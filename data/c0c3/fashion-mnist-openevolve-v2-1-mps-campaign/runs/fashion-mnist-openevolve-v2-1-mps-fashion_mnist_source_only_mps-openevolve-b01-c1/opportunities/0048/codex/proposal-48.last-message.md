MECHANISM: Hierarchical flip-pair supervision

HYPOTHESIS: Supervising flip-averaged pairs as an intermediate aggregation level will exceed 9,239 correct predictions by aligning training with the flip-paired validation ensemble while retaining most early individual-view learning.

INTENDED_EDIT: Reallocate one quarter of the individual-view loss to cross-entropy on three flip-averaged view pairs, preserving the proven six-view ensemble loss and cosine curriculum.

EVIDENCE: The cosine ensemble curriculum achieved the best 9,239-correct result, while architecture, head, smoothing, and EMA changes regressed; this motivates a compute-light refinement of the successful aggregation objective.

<<<<<<< SEARCH
    individual_loss = (
        0.9 * full_individual_loss + 0.1 * central_individual_loss
    )
    ensemble_loss = F.cross_entropy(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=0.02,
    )
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
    )
=======
    individual_loss = (
        0.9 * full_individual_loss + 0.1 * central_individual_loss
    )
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
    pair_loss = 0.9 * full_pair_loss + 0.1 * central_pair_loss
    view_loss = 0.75 * individual_loss + 0.25 * pair_loss
    ensemble_loss = F.cross_entropy(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=0.02,
    )
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * view_loss
        + ensemble_weight * ensemble_loss
    )
>>>>>>> REPLACE