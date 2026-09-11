MECHANISM: Validation-weight-matched center anchoring

HYPOTHESIS: Retaining the validated 37.5% cross-offset rate while center-anchoring 71.37% of those pairs will exceed 9,325 correct predictions by matching training-view exposure to the inference ensemble’s measured 1.5578× center reliability.

INTENDED_EDIT: Restore the best paired-view training design, but construct a calibrated mixture of random and center-to-side cross-offset pairs whose marginal offset frequencies match validation pooling weights.

EVIDENCE: Reference Design 1 achieved the best result—9,325 correct—at 37.5% cross-offset exposure, and its validated inference ensemble weights the center offset 1.5578× more than each side offset; this patch preserves the successful dose while aligning training exposure with that reliability signal.

<<<<<<< SEARCH
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    paired_logits = model(torch.cat((images, images.flip(-1)), dim=0))
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)

    classification_loss = 0.5 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=0.02,
        )
        + F.cross_entropy(
            flipped_logits,
            labels,
            label_smoothing=0.02,
        )
    )

    original_log_probabilities = F.log_softmax(original_logits, dim=-1)
    flipped_log_probabilities = F.log_softmax(flipped_logits, dim=-1)
    consistency_loss = 0.5 * (
        F.kl_div(
            original_log_probabilities,
            flipped_log_probabilities.exp(),
            reduction="batchmean",
        )
        + F.kl_div(
            flipped_log_probabilities,
            original_log_probabilities.exp(),
            reduction="batchmean",
        )
    )
    return classification_loss + 0.05 * consistency_loss
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    batch, channels, height, width = images.shape
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    positions = torch.randint(0, 5, (batch,), device=images.device)
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