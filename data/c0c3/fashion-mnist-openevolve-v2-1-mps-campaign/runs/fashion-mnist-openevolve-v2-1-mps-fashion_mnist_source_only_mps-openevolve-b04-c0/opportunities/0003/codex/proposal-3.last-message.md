MECHANISM: Mild translation-invariant augmentation

HYPOTHESIS: Adding random integer translations of up to two pixels while retaining batch size 128 will exceed 9,283 correct predictions by improving spatial robustness without reducing optimizer-update density.

INTENDED_EDIT: Extend the validated horizontal-flip augmentation with per-image, border-padded random translations.

EVIDENCE: The 216,538-parameter batch-128 design reached 92.83%, while batch 64 reduced correct predictions to 9,272; this motivates preserving the successful optimization regime and testing a targeted augmentation improvement.

<<<<<<< SEARCH
    del step, total_steps
    flip_mask = torch.rand(
        images.shape[0], 1, 1, 1, device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
=======
    del step, total_steps
    batch, _, height, width = images.shape
    shifts = torch.randint(
        -2, 3, (batch, 2), device=images.device
    )
    batch_index = torch.arange(
        batch, device=images.device
    )[:, None, None]
    y_index = (
        torch.arange(height, device=images.device)[None, :, None]
        - shifts[:, 0, None, None]
    ).clamp_(0, height - 1)
    x_index = (
        torch.arange(width, device=images.device)[None, None, :]
        - shifts[:, 1, None, None]
    ).clamp_(0, width - 1)
    images = images[:, 0][batch_index, y_index, x_index].unsqueeze(1)
    flip_mask = torch.rand(
        batch, 1, 1, 1, device=images.device
    ) < 0.5
    images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
>>>>>>> REPLACE