MECHANISM: Early cyclic translation augmentation

HYPOTHESIS: Training on balanced one-pixel translations during the first half will exceed 9,328 correct predictions by improving spatial robustness while leaving late optimization and EMA collection clean.

INTENDED_EDIT: Replace the functionally redundant random horizontal flip with a deterministic cycle through all nine one-pixel translations during the first half of training.

EVIDENCE: Annealing dropout and label smoothing improved validation correct from 9,316 to 9,328, supporting early-only regularization; the prior translation verification timed out, so this low-cost hypothesis remains unresolved.

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
    if progress < 0.50:
        phase = step % 9
        offset_y = phase // 3 - 1
        offset_x = phase % 3 - 1
        padded = F.pad(images, (1, 1, 1, 1))
        images = padded[
            :,
            :,
            1 + offset_y:29 + offset_y,
            1 + offset_x:29 + offset_x,
        ]
    return images, labels
>>>>>>> REPLACE