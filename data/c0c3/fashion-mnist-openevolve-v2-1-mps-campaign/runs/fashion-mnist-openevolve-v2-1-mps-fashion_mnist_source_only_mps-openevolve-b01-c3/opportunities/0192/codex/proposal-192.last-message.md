MECHANISM: Quadratic-interpolated hybrid calibration

HYPOTHESIS: Restoring the validated hybrid ensemble and setting its calibration scale to 1.24739 will preserve 9,325 correct predictions while reducing cross-entropy below 0.1929897728.

INTENDED_EDIT: Restore uniform 37.5% paired-offset consistency training and 12.5% probability-blended inference, then move the calibration scale to the estimated optimum from the three verified calibration points.

EVIDENCE: Scales 1.22775, 1.24003, and 1.24623 produced progressively lower cross-entropies while retaining exactly 9,325 correct predictions; quadratic interpolation places the local minimum near 1.24739.

<<<<<<< SEARCH
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        calibration = 1.22775 * (
            0.92211476 + 0.07788524 * view_agreement
        )
        return calibration.unsqueeze(1) * pooled_logits
=======
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        offset_probabilities = view_logits.softmax(dim=-1).mean(dim=1)
        pooled_probabilities = (
            offset_weights.unsqueeze(-1) * offset_probabilities
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        hybrid_probabilities = (
            0.875 * pooled_logits.softmax(dim=-1)
            + 0.125 * pooled_probabilities
        )
        pooled_logits = hybrid_probabilities.clamp_min(1.0e-8).log()
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
        agreement_uncertainty = view_agreement * (1.0 - view_agreement)
        calibration = 1.24739 * (
            0.92211476
            + 0.07788524 * view_agreement
            - 0.02 * agreement_uncertainty
        )
        return calibration.unsqueeze(1) * pooled_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    batch, channels, height, width = images.shape
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    positions = torch.randint(0, 5, (batch,), device=images.device)
    row_offsets = torch.tensor(
        (1, 0, 2, 1, 1), device=images.device
    )[positions].unsqueeze(1)
    col_offsets = torch.tensor(
        (1, 1, 1, 0, 2), device=images.device
    )[positions].unsqueeze(1)

    rows = row_offsets + torch.arange(height, device=images.device).unsqueeze(0)
    row_index = rows[:, None, :, None].expand(
        batch, channels, height, padded.shape[-1]
    )
    images = padded.gather(2, row_index)

    cols = col_offsets + torch.arange(width, device=images.device).unsqueeze(0)
    col_index = cols[:, None, None, :].expand(batch, channels, height, width)
    images = images.gather(3, col_index)

    flip_mask = torch.rand(batch, device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    return images, labels
=======
    del step, total_steps
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    del step, total_steps
    batch, channels, height, width = images.shape
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    positions = torch.randint(0, 5, (batch,), device=images.device)
    alternate_positions = (
        positions
        + torch.randint(1, 5, (batch,), device=images.device)
    ) % 5
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.375
    partner_positions = torch.where(
        cross_offset_mask,
        alternate_positions,
        positions,
    )

    row_choices = torch.tensor(
        (1, 0, 2, 1, 1),
        device=images.device,
    )
    col_choices = torch.tensor(
        (1, 1, 1, 0, 2),
        device=images.device,
    )

    def crop_views(selected_positions: torch.Tensor) -> torch.Tensor:
        row_offsets = row_choices[selected_positions].unsqueeze(1)
        col_offsets = col_choices[selected_positions].unsqueeze(1)
        rows = (
            row_offsets
            + torch.arange(height, device=images.device).unsqueeze(0)
        )
        row_index = rows[:, None, :, None].expand(
            batch,
            channels,
            height,
            padded.shape[-1],
        )
        cropped = padded.gather(2, row_index)
        cols = (
            col_offsets
            + torch.arange(width, device=images.device).unsqueeze(0)
        )
        col_index = cols[:, None, None, :].expand(
            batch,
            channels,
            height,
            width,
        )
        return cropped.gather(3, col_index)

    original_views = crop_views(positions)
    partner_views = crop_views(partner_positions)
    flip_mask = torch.rand(batch, device=images.device) < 0.5
    original_views = torch.where(
        flip_mask[:, None, None, None],
        original_views.flip(-1),
        original_views,
    )
    partner_views = torch.where(
        flip_mask[:, None, None, None],
        partner_views,
        partner_views.flip(-1),
    )

    paired_logits = model(
        torch.cat((original_views, partner_views), dim=0)
    )
    original_logits, partner_logits = paired_logits.chunk(2, dim=0)
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

    original_log_probabilities = F.log_softmax(original_logits, dim=-1)
    partner_log_probabilities = F.log_softmax(partner_logits, dim=-1)
    consistency_loss = 0.5 * (
        F.kl_div(
            original_log_probabilities,
            partner_log_probabilities.exp(),
            reduction="none",
        ).sum(dim=-1)
        + F.kl_div(
            partner_log_probabilities,
            original_log_probabilities.exp(),
            reduction="none",
        ).sum(dim=-1)
    )
    return classification_loss + 0.05 * consistency_loss.mean()
>>>>>>> REPLACE