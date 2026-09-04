MECHANISM: Controlled one-pixel translation augmentation

HYPOTHESIS: Training on centered images half the time and cardinally shifted images otherwise will exceed 9,319 correct predictions by reducing the flattening classifier’s sensitivity to small positional changes without excessive augmentation.

INTENDED_EDIT: Apply a deterministic eight-step augmentation cycle containing four centered batches and one batch for each one-pixel cardinal translation.

EVIDENCE: The strongest configuration reached 9,319 correct, while repeated local changes to smoothing, dropout, and scheduling failed; this tests an orthogonal input-side improvement while preserving the verified model and optimizer.

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
    phase = step % 8
    if phase < 4:
        return images, labels

    offset_y, offset_x = (
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2),
    )[phase - 4]
    padded = F.pad(images, (1, 1, 1, 1))
    images = padded[
        :,
        :,
        offset_y : offset_y + 28,
        offset_x : offset_x + 28,
    ]
    return images, labels
>>>>>>> REPLACE