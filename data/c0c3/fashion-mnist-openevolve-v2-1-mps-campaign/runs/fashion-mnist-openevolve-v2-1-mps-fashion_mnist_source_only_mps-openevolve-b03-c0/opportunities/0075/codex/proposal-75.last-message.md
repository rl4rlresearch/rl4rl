MECHANISM: Boundary-safe mixed translation augmentation

HYPOTHESIS: Training each batch with half original images and half reflection-padded one-pixel translations will exceed 9,319 correct predictions by learning mild translation invariance without blank-edge artifacts.

INTENDED_EDIT: Replace the identity batch preparation with deterministic, direction-cycling one-pixel translations applied to alternating examples; keep labels, exposure, architecture, optimizer, and evaluation calibration unchanged.

EVIDENCE: Equal-weight translation ensembling lost only 19 correct predictions and implicated shifted-boundary artifacts, while horizontal-reflection ensembling lost 861; this indicates translation is the more compatible invariance and motivates learning it with artifact-resistant padding.

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
    height, width = images.shape[-2:]
    padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
    crop_y, crop_x = ((0, 1), (2, 1), (1, 0), (1, 2))[step % 4]
    augmented = images.clone()
    augmented[1::2] = padded[
        1::2, :, crop_y : crop_y + height, crop_x : crop_x + width
    ]
    return augmented, labels
>>>>>>> REPLACE