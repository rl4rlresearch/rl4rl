MECHANISM: Early random translation augmentation with clean-data fine-tuning

HYPOTHESIS: Per-image one-pixel translations during the first 75% of training, followed by centered-image fine-tuning, will increase validation_correct above 9,359 without changing evaluation-time calibration.

INTENDED_EDIT: Apply random replicate-padded ±1-pixel crops in `prepare_training_batch` for the first three quarters of training, while preserving centered batches for the final quarter.

EVIDENCE: Equal-weight translation test-time ensembling reduced validation_correct from 9,359 to 9,348, showing that shifted views are not currently modeled robustly; learning limited translation invariance before a clean-data finishing phase directly tests whether that sensitivity can be reduced without averaging shifted logits at evaluation.

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
    if step < (3 * total_steps) // 4:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        windows = padded.unfold(2, 28, 1).unfold(3, 28, 1)
        batch_indices = torch.arange(images.shape[0], device=images.device)
        y_offsets = torch.randint(0, 3, (images.shape[0],), device=images.device)
        x_offsets = torch.randint(0, 3, (images.shape[0],), device=images.device)
        images = windows[
            batch_indices, :, y_offsets, x_offsets
        ].contiguous()
    return images, labels
>>>>>>> REPLACE