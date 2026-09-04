MECHANISM: Early cyclic translation augmentation

HYPOTHESIS: Replacing the functionally redundant random flip with one-pixel translations during the first half of training will exceed 9,328 correct predictions by adding spatial regularization while preserving clean late optimization and EMA collection.

INTENDED_EDIT: Cycle deterministically through all nine one-pixel translations during the first half of training, then use unmodified images during the second half.

EVIDENCE: Annealing dropout and label smoothing during the final half improved validation correct from 9,316 to 9,328, supporting early-only regularization; meanwhile, random flipping is redundant because the model already symmetrically fuses every image with its mirrored view.

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
        height, width = images.shape[-2:]
        phase = step % 9
        offset_y = phase // 3
        offset_x = phase % 3
        images = F.pad(images, (1, 1, 1, 1), mode="replicate")
        images = images[
            ...,
            offset_y : offset_y + height,
            offset_x : offset_x + width,
        ]
    return images, labels
>>>>>>> REPLACE