MECHANISM: Compute-efficient deterministic translation augmentation

HYPOTHESIS: Batch-shared one-pixel translations on 8 of every 17 steps will exceed 9,330 correct predictions by improving translation invariance while avoiding the per-image augmentation overhead that previously timed out.

INTENDED_EDIT: Cycle deterministically through identity and eight one-pixel translations, using replicate padding and one shared transform per batch.

EVIDENCE: The verified baseline reaches 9,330 correct without translation augmentation; the prior per-image translation attempt targeted a distinct useful invariance but did not finish, motivating an equivalent 47% exposure rate with substantially less augmentation overhead.

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
    cycle_position = step % 17
    if cycle_position >= 9:
        shifts = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
        dy, dx = shifts[cycle_position - 9]
        height, width = images.shape[-2:]
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        images = padded[
            ...,
            1 - dy : 1 - dy + height,
            1 - dx : 1 - dx + width,
        ]
    return images, labels
>>>>>>> REPLACE