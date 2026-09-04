MECHANISM: Conservative per-image translation augmentation

HYPOTHESIS: Applying one-pixel translations to approximately 47% of training examples will exceed 9,330 correct predictions by improving local translation invariance without adding model forwards or parameters.

INTENDED_EDIT: Randomly select identity or one of eight one-pixel crops for each training image, with identity retained approximately 53% of the time.

EVIDENCE: The best verified design already achieves 9,330 correct with calibrated flip ensembling, while subsequent evaluation-fusion and loss modifications produced no completed contrary metrics; a lightweight spatial augmentation targets a distinct remaining invariance without changing the verified architecture or objective.

<<<<<<< SEARCH
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    return images, labels
=======
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    crops = F.unfold(padded, kernel_size=(28, 28))
    choices = torch.randint(0, 17, (images.size(0),), device=images.device)
    choices = torch.where(choices < 9, choices, torch.full_like(choices, 4))
    images = crops.gather(
        2,
        choices[:, None, None].expand(-1, crops.size(1), 1),
    ).squeeze(2).view_as(images)
    return images, labels
>>>>>>> REPLACE