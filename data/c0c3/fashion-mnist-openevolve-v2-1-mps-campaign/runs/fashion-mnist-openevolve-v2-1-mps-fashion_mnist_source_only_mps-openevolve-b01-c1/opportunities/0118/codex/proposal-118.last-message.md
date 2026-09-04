MECHANISM: Low-strength batch Mixup

HYPOTHESIS: Mixing each training image with a randomly paired example using Beta(0.2, 0.2) weights will exceed 9,257 correct predictions by regularizing class boundaries without introducing the harmful geometric distortions observed with rotation.

INTENDED_EDIT: Apply batch-level Mixup before generating the existing six views and replace each supervised cross-entropy term with the corresponding mixed-target loss while retaining fixed 0.02 label smoothing.

EVIDENCE: Rotation regressed to 9,201 and evaluation-matched crop sampling reached only 9,255, while the unchanged architecture remains best; this motivates testing non-geometric vicinal augmentation while preserving the proven view pipeline and fixed smoothing.

<<<<<<< SEARCH
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    offsets = torch.randint(0, 5, (2,))
=======
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    concentration = images.new_tensor(0.2)
    mix_weight = torch.distributions.Beta(
        concentration, concentration
    ).sample()
    mix_weight = torch.maximum(mix_weight, 1.0 - mix_weight)
    permutation = torch.randperm(images.shape[0], device=images.device)
    secondary_labels = labels[permutation]
    images = (
        mix_weight * images
        + (1.0 - mix_weight) * images[permutation]
    )

    def supervised_loss(
        predictions: torch.Tensor,
        repeat_count: int,
    ) -> torch.Tensor:
        primary_loss = F.cross_entropy(
            predictions,
            labels.repeat(repeat_count),
            label_smoothing=0.02,
        )
        secondary_loss = F.cross_entropy(
            predictions,
            secondary_labels.repeat(repeat_count),
            label_smoothing=0.02,
        )
        return (
            mix_weight * primary_loss
            + (1.0 - mix_weight) * secondary_loss
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
    full_individual_loss = supervised_loss(
        torch.cat((logits_a, logits_b, flipped_a, flipped_b), dim=0),
        4,
    )
    central_individual_loss = supervised_loss(
        torch.cat((central_logits, flipped_central), dim=0),
        2,
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
    full_pair_loss = supervised_loss(
        torch.cat(
            (
                0.5 * (logits_a + flipped_a),
                0.5 * (logits_b + flipped_b),
            ),
            dim=0,
        ),
        2,
    )
    central_pair_loss = supervised_loss(
        0.5 * (central_logits + flipped_central),
        1,
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
    ensemble_loss = supervised_loss(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        1,
    )
>>>>>>> REPLACE