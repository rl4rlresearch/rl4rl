MECHANISM: Hierarchical antithetic-crop aggregation

HYPOTHESIS: Allocating 25% of the full-view pair objective to the aggregate of both antithetic crop pairs will exceed 9,247 correct predictions by teaching crop-level complementarity that matches validation-time crop averaging.

INTENDED_EDIT: Add cross-entropy on the four-logit antithetic crop aggregate and blend it into the existing full flip-pair loss, preserving all architecture, augmentation, schedules, and final ensemble supervision.

EVIDENCE: Flip-pair hierarchy improved correctness from 9,239 to 9,245 and its cosine curriculum reached 9,247, while further schedule-shape refinements did not improve; this motivates extending the successful hierarchy to the currently unsupervised intermediate aggregation across full-crop pairs.

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
    pair_loss = 0.9 * full_pair_loss + 0.1 * central_pair_loss
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
        label_smoothing=0.02,
    )
    full_crop_aggregate_loss = F.cross_entropy(
        0.25 * (logits_a + logits_b + flipped_a + flipped_b),
        labels,
        label_smoothing=0.02,
    )
    full_hierarchical_loss = (
        0.75 * full_pair_loss + 0.25 * full_crop_aggregate_loss
    )
    central_pair_loss = F.cross_entropy(
        0.5 * (central_logits + flipped_central),
        labels,
        label_smoothing=0.02,
    )
    pair_loss = (
        0.9 * full_hierarchical_loss + 0.1 * central_pair_loss
    )
>>>>>>> REPLACE