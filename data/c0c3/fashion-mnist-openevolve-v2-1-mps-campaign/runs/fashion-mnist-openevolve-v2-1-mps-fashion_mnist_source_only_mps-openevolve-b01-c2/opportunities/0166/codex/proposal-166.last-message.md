MECHANISM: Sparse one-pixel translation augmentation

HYPOTHESIS: Translating every fourth batch during the first 75% of training will exceed 9,331 correct predictions while remaining within the verification time limit.

INTENDED_EDIT: Cycle deterministically through eight one-pixel translations on 18.75% of all training batches, leaving the final quarter and all other batches clean.

EVIDENCE: The denser translation experiment timed out without accuracy evidence, while inference fusion and calibration changes saturated at 9,331 correct; sparsifying the augmentation tests the same spatial-robustness hypothesis with substantially lower overhead.

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
    if step < (3 * total_steps) // 4 and step % 4 == 0:
        directions = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
        shift = directions[(step // 4) % len(directions)]
        images = torch.roll(images, shifts=shift, dims=(-2, -1))
    return images, labels
>>>>>>> REPLACE