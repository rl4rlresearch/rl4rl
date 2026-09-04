MECHANISM: Low-overhead second-pass horizontal-flip augmentation

HYPOTHESIS: Flipping half of each second-exposure batch will exceed 9,281 correct predictions by improving left-right invariance while avoiding the runtime cost of padded translation augmentation.

INTENDED_EDIT: Preserve the first exposure unchanged, then deterministically flip alternating images horizontally during the second exposure; retain the best architecture, smoothing, schedule, and evaluation calibration.

EVIDENCE: Architectural channel-mixing variants underperformed the 9,281-correct baseline, while second-pass translation augmentation timed out; horizontal flipping tests the same orthogonal invariance strategy with substantially less preprocessing.

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
    if step >= total_steps // 2:
        images = images.clone()
        images[::2] = torch.flip(images[::2], dims=(-1,))
    return images, labels
>>>>>>> REPLACE