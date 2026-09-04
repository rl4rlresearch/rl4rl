MECHANISM: Annealed batch-shared cardinal translation

HYPOTHESIS: Replacing the symmetry-redundant flips with one-pixel cardinal translations during the first half of training will exceed 9,328 correct predictions by improving alignment tolerance while retaining a fully clean second half.

INTENDED_EDIT: Cycle whole batches through centered and four one-pixel translated views before EMA collection, using replicate padding and removing the redundant random flips.

EVIDENCE: The model is exactly invariant to horizontal flips, while prior translation experiments timed out without resolving their accuracy benefit; this tests translation tolerance with a lightweight batch-shared operation that runs on only two-fifths of all training steps.

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
    if 2 * step < total_steps:
        offsets = (
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        )
        shift_y, shift_x = offsets[step % len(offsets)]
        if shift_y != 0 or shift_x != 0:
            padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
            images = padded[
                :,
                :,
                1 + shift_y : 29 + shift_y,
                1 + shift_x : 29 + shift_x,
            ].contiguous()
    return images, labels
>>>>>>> REPLACE