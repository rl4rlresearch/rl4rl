MECHANISM: Late-ramped cross-offset consistency

HYPOTHESIS: Keeping 25% cross-offset reflected pairs but ramping their consistency weight from 0 to 5% will exceed 9,312 correct predictions by avoiding noisy early alignment while preserving the validated average 2.5% regularization.

INTENDED_EDIT: Move stochastic offset construction into the loss, retain two supervised views per example, and cosine-ramp only the cross-offset consistency coefficient while exact-reflection pairs remain at 5%.

EVIDENCE: Static 25% cross-offset pairing retained 9,312 correct and reduced cross-entropy from 0.196252 to 0.193848; scheduling the same average cross-offset consistency is a compute-neutral test of whether early invariance pressure prevented a correctness gain.

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
    batch, channels, height, width = images.shape
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    positions = torch.randint(0, 5, (batch,), device=images.device)
    alternate_positions = (
        positions
        + torch.randint(1, 5, (batch,), device=images.device)
    ) % 5
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.25
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

    paired_logits = model(torch.cat((original_views, partner_views), dim=0))
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

    progress = min((step + 1) / max(total_steps, 1), 1.0)
    cross_offset_weight = 0.025 * (
        1.0 - math.cos(math.pi * progress)
    )
    consistency_weights = torch.where(
        cross_offset_mask,
        consistency_loss.new_tensor(cross_offset_weight),
        consistency_loss.new_tensor(0.05),
    )
    return classification_loss + (
        consistency_weights * consistency_loss
    ).mean()
>>>>>>> REPLACE