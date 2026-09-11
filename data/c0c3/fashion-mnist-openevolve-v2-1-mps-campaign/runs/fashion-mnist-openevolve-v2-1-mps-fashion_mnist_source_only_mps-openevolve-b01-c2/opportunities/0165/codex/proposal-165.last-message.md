MECHANISM: Conservative translation augmentation with clean late training

HYPOTHESIS: Applying one-pixel translations to half of each batch during the first 75% of training, followed by clean-only optimization, will exceed 9,331 correct predictions by improving spatial robustness without creating a persistent train-validation mismatch.

INTENDED_EDIT: Deterministically cycle through eight one-pixel translation directions for alternating examples, preserving half of every batch unchanged and disabling augmentation for the final quarter of training.

EVIDENCE: Inference-only fusion and calibration refinements have saturated at 9,331 correct, while asymmetric training, EMA, and probability fusion reduced correctness; this motivates an orthogonal, low-overhead augmentation that leaves the verified model, objective, and inference rule intact.

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
    if step < 3 * total_steps // 4:
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
        shift_y, shift_x = directions[step % len(directions)]
        augmented = images.clone()
        padded = F.pad(images[1::2], (1, 1, 1, 1))
        augmented[1::2] = padded[
            :,
            :,
            1 + shift_y : 1 + shift_y + images.shape[-2],
            1 + shift_x : 1 + shift_x + images.shape[-1],
        ]
        images = augmented
    return images, labels
>>>>>>> REPLACE