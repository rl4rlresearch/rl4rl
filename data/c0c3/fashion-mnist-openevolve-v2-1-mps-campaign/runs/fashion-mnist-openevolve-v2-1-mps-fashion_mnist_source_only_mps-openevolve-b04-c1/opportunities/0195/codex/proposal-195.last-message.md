MECHANISM: Per-example integer translation augmentation

HYPOTHESIS: Training on independently translated crops will exceed 9,330 correct predictions by regularizing the proven position-sensitive classifier against small alignment changes without discarding spatial layout.

INTENDED_EDIT: Add replicate-padded random translations of up to two pixels per axis to every training image while preserving batch shape and the existing flip handling.

EVIDENCE: Global pooling fell to 9,290 correct and widening the spatial classifier fell to 9,301, indicating that spatial layout is valuable but additional classifier capacity is ineffective; translation augmentation preserves that layout while reducing brittle position dependence.

<<<<<<< SEARCH
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
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
    padding = 2
    batch_size, channels, height, width = images.shape
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    offsets = torch.randint(
        0,
        2 * padding + 1,
        (batch_size, 2),
        device=images.device,
    )
    row_indices = (
        offsets[:, 0, None, None, None]
        + torch.arange(height, device=images.device)[None, None, :, None]
    ).expand(-1, channels, -1, padded.shape[-1])
    images = padded.gather(2, row_indices)
    column_indices = (
        offsets[:, 1, None, None, None]
        + torch.arange(width, device=images.device)[None, None, None, :]
    ).expand(-1, channels, height, -1)
    images = images.gather(3, column_indices)

    flip_mask = torch.rand(batch_size, device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
    return images, labels
>>>>>>> REPLACE