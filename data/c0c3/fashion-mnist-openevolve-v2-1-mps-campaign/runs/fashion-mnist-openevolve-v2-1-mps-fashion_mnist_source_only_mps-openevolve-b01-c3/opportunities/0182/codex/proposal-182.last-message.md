MECHANISM: Pair-ensemble-aligned classification loss

HYPOTHESIS: Restoring uniformly sampled 37.5% cross-offset pairs and assigning 12.5% of classification supervision to the pair-averaged logits will exceed 9,325 correct predictions by aligning training with the successful logit-averaged inference ensemble.

INTENDED_EDIT: Remove the detrimental center anchoring, restore Reference Design 1’s cross-offset sampling, and blend individual-view cross-entropy with cross-entropy on each pair’s mean logits while retaining 5% consistency regularization.

EVIDENCE: Uniform 37.5% cross-offset training achieved the best verified result of 9,325 correct, while center anchoring reduced it to 9,324. The best design averages transformed-view logits at inference, motivating a conservative train-time objective on paired mean logits.

<<<<<<< SEARCH
    random_alternate_positions = (
        positions
        + torch.randint(1, 5, (batch,), device=images.device)
    ) % 5
    center_alternate_positions = torch.where(
        positions == 0,
        torch.randint(1, 5, (batch,), device=images.device),
        torch.zeros_like(positions),
    )
    center_anchor_mask = torch.rand(batch, device=images.device) < 0.7137
    alternate_positions = torch.where(
        center_anchor_mask,
        center_alternate_positions,
        random_alternate_positions,
    )
=======
    alternate_positions = (
        positions
        + torch.randint(1, 5, (batch,), device=images.device)
    ) % 5
>>>>>>> REPLACE

<<<<<<< SEARCH
    classification_loss = 0.5 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=0.02,
        )
        + F.cross_entropy(
            partner_logits,
            labels,
            label_smoothing=0.02,
        )
    )
=======
    individual_classification_loss = 0.5 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=0.02,
        )
        + F.cross_entropy(
            partner_logits,
            labels,
            label_smoothing=0.02,
        )
    )
    ensemble_classification_loss = F.cross_entropy(
        0.5 * (original_logits + partner_logits),
        labels,
        label_smoothing=0.02,
    )
    classification_loss = (
        0.875 * individual_classification_loss
        + 0.125 * ensemble_classification_loss
    )
>>>>>>> REPLACE