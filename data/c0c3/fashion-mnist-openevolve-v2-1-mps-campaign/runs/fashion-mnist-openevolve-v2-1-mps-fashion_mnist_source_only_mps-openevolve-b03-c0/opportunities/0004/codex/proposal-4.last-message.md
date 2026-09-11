MECHANISM: Translation-and-reflection invariance through lightweight augmentation

HYPOTHESIS: Adding small random translations and per-image horizontal flips while retaining the verified 237,346-parameter model will exceed 9,257 correct predictions by reducing overfitting to image position and orientation.

INTENDED_EDIT: Augment each training batch with a randomly shifted 28×28 crop from two-pixel replicated padding and independent horizontal flips.

EVIDENCE: The verified residual model reached 92.57% accuracy, while reallocating capacity to another residual block reduced accuracy to 92.50%; this motivates improving generalization without changing the successful architecture or optimizer.

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
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    top = int(torch.randint(0, 5, ()).item())
    left = int(torch.randint(0, 5, ()).item())
    images = padded[:, :, top : top + 28, left : left + 28]
    flip_mask = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
>>>>>>> REPLACE