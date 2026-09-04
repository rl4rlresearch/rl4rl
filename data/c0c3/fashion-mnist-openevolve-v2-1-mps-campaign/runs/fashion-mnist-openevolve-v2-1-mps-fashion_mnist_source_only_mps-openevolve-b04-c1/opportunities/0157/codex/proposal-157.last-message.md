MECHANISM: Annealed per-image translation augmentation

HYPOTHESIS: Training with ±2-pixel translations early, reducing to ±1 and then clean images, will exceed 9,328 correct predictions by improving local shift tolerance without sacrificing terminal alignment to the fixed validation distribution.

INTENDED_EDIT: Add inexpensive per-example translated crops during the first 75% of training, annealing their radius from two pixels to one before disabling them for the final quarter.

EVIDENCE: Global pooling reduced correctness to 9,290, showing that spatial layout must be preserved, while added spatial refinement timed out; annealed translation regularizes the existing position-sensitive representation without adding learned parameters or inference computation.

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
    progress = min(step / max(total_steps, 1), 1.0)
    translation_radius = 2 if progress < 0.50 else (
        1 if progress < 0.75 else 0
    )
    if translation_radius:
        padded = F.pad(
            images,
            (translation_radius,) * 4,
            mode="replicate",
        )
        crop_range = 2 * translation_radius + 1
        row_offsets = torch.randint(
            crop_range,
            (images.shape[0],),
            device=images.device,
        )
        column_offsets = torch.randint(
            crop_range,
            (images.shape[0],),
            device=images.device,
        )
        coordinates = torch.arange(28, device=images.device)
        rows = row_offsets[:, None] + coordinates[None, :]
        images = padded.gather(
            2,
            rows[:, None, :, None].expand(
                -1, padded.shape[1], -1, padded.shape[3]
            ),
        )
        columns = column_offsets[:, None] + coordinates[None, :]
        images = images.gather(
            3,
            columns[:, None, None, :].expand(
                -1, images.shape[1], images.shape[2], -1
            ),
        )

    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
    return images, labels
>>>>>>> REPLACE