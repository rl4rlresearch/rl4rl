MECHANISM: Low-overhead cyclic one-pixel translation augmentation

HYPOTHESIS: Replacing the redundant preparation flip with centered-or-cardinal one-pixel translations will exceed 9,322 correct predictions by improving shift robustness while retaining the faster hard-maximum attention model.

INTENDED_EDIT: Use the unmodified batch on half the steps and a shared replicate-padded one-pixel translation on the remainder, avoiding costly per-example crop indexing.

EVIDENCE: Hard-maximum attention reached 9,320 correct within the time limit, while per-example translations on the 9,322-correct top-four model timed out; the existing preparation flip is redundant because training_loss already evaluates both orientations.

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
    del total_steps
    phase = step % 8
    if phase < 4:
        return images, labels

    offsets = ((0, 1), (2, 1), (1, 0), (1, 2))
    y_offset, x_offset = offsets[phase - 4]
    padded_images = F.pad(images, (1, 1, 1, 1), mode="replicate")
    images = padded_images[
        ...,
        y_offset : y_offset + images.shape[-2],
        x_offset : x_offset + images.shape[-1],
    ]
    return images, labels
>>>>>>> REPLACE