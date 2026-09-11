MECHANISM: Sparse one-pixel translation augmentation

HYPOTHESIS: Translating 25% of training images by one reflected pixel will improve spatial robustness and produce more than 9,319 correct validation predictions.

INTENDED_EDIT: Randomly replace one quarter of each training batch with a uniformly selected one-pixel translation matching the four evaluation-time views.

EVIDENCE: Evaluation-time translated logits consistently reduced cross-entropy while preserving all 9,319 predictions; with the blend now locally saturated at 30%, training on those same perturbations is the most direct test of whether their complementary signal can improve correctness.

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
    padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
    translated_views = torch.stack(
        (
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
        ),
        dim=0,
    )
    batch_indices = torch.arange(images.shape[0], device=images.device)
    directions = torch.randint(4, (images.shape[0],), device=images.device)
    translated_images = translated_views[directions, batch_indices]
    translate_mask = (
        torch.rand(images.shape[0], 1, 1, 1, device=images.device) < 0.25
    )
    images = torch.where(translate_mask, translated_images, images)
    return images, labels
>>>>>>> REPLACE