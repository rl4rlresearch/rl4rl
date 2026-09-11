MECHANISM: Early per-image translation augmentation with clean late fine-tuning

HYPOTHESIS: Applying random translations of up to two pixels during the first 75% of training will exceed 9,328 correct predictions by improving positional robustness while the final clean-data phase preserves precise classification boundaries.

INTENDED_EDIT: Add efficient replicate-padded per-image random crops before the existing horizontal flip, disabling translation augmentation for the final quarter of training.

EVIDENCE: The unchanged early-fusion architecture remains best at 9,328 correct, while structural head and refinement changes failed or timed out; augmentation tests a distinct, parameter-neutral source of generalization and replaces some value lost because the existing flip is largely redundant with symmetric paired-view fusion.

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
    if progress < 0.75:
        batch_size, _, height, width = images.shape
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")[:, 0]
        offset_y = torch.randint(0, 5, (batch_size,), device=images.device)
        offset_x = torch.randint(0, 5, (batch_size,), device=images.device)
        batch_indices = torch.arange(batch_size, device=images.device)
        rows = offset_y[:, None] + torch.arange(height, device=images.device)
        columns = offset_x[:, None] + torch.arange(width, device=images.device)
        images = padded[
            batch_indices[:, None, None],
            rows[:, :, None],
            columns[:, None, :],
        ].unsqueeze(1)

    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
    return images, labels
>>>>>>> REPLACE