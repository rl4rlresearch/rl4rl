MECHANISM: Light translation-and-reflection training augmentation

HYPOTHESIS: Adding mild random translations and per-image horizontal reflections during training will increase validation_correct above 9,258 without evaluation-time overhead.

INTENDED_EDIT: Augment half of training batches with a shared ±1-pixel translation and independently reflect each image with 50% probability; retain the verified model, optimizer, and 1.24 calibration.

EVIDENCE: Logit calibration repeatedly preserved exactly 9,258 predictions, so further scaling only targets the tie-breaker; the horizontal-reflection evaluation ensemble targeted correctness but timed out, motivating invariance learning during training without doubling evaluation compute.

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
    del step, total_steps

    if torch.rand(()).item() < 0.5:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        top = int(torch.randint(0, 3, ()).item())
        left = int(torch.randint(0, 3, ()).item())
        images = padded[:, :, top : top + 28, left : left + 28]

    flip_mask = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
>>>>>>> REPLACE