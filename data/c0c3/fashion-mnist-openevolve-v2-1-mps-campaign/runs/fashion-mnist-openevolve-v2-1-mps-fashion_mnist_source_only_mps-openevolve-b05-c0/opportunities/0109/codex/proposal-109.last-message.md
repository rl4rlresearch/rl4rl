MECHANISM: Balanced one-pixel translation augmentation

HYPOTHESIS: Training on deterministic one-pixel translations for half the minibatches will increase validation_correct above 9,258 by improving local spatial robustness without imposing reflection invariance.

INTENDED_EDIT: Apply balanced up, down, left, and right translations to alternating training batches using replicated border padding.

EVIDENCE: Evaluation-logit scaling repeatedly preserved exactly 9,258 predictions, so further calibration cannot improve the primary metric. Horizontal reflection reduced validation_correct to 8,883, motivating a milder, label-preserving spatial augmentation.

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
    if step % 2 == 0:
        shift_y, shift_x = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        )[(step // 2) % 4]
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        images = padded[
            :,
            :,
            1 - shift_y : 29 - shift_y,
            1 - shift_x : 29 - shift_x,
        ].contiguous()
    return images, labels
>>>>>>> REPLACE