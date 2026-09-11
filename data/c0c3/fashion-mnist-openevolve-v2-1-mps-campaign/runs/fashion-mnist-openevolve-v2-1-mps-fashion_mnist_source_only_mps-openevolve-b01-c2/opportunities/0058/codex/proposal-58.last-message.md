MECHANISM: Deterministic one-pixel translation augmentation

HYPOTHESIS: Training on centered images plus four one-pixel translations will exceed 9,330 correct predictions by improving local shift invariance while preserving the verified architecture, optimization, flip ensemble, and calibration.

INTENDED_EDIT: Cycle batches through centered, up, down, left, and right one-pixel crops using zero padding, without increasing parameter count or model-forward cost.

EVIDENCE: Symmetric flip training and inference achieved the best verified result of 9,330 correct, while altered fusion, EMA, and smoothing reduced correctness; this motivates preserving the proven procedure and adding a lightweight spatial-invariance augmentation.

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
    del total_steps
    offsets = (
        (1, 1),
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2),
    )
    top, left = offsets[step % len(offsets)]
    padded = F.pad(images, (1, 1, 1, 1))
    images = padded[:, :, top : top + 28, left : left + 28]
    return images, labels
>>>>>>> REPLACE