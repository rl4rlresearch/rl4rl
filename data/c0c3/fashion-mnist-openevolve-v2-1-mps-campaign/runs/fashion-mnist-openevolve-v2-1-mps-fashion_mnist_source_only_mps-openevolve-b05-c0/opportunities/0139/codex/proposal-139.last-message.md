MECHANISM: Reflection-padded random translation augmentation

HYPOTHESIS: Training with independent random translations of up to two pixels will increase validation_correct above 9,260 by improving local translation invariance without imposing the harmful reflection invariance observed previously.

INTENDED_EDIT: Replace the unchanged training inputs with vectorized per-image random crops from two-pixel reflection padding; preserve the verified optimizer, EMA, BatchNorm interpolation, label smoothing, and evaluation calibration.

EVIDENCE: Horizontal-reflection augmentation reduced validation_correct from 9,260 to 9,223, showing that orientation-changing invariance is harmful; small translations are a less semantic, orthogonal augmentation aimed at accuracy rather than calibration.

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    padding = 2
    padded = F.pad(images, (padding, padding, padding, padding), mode="reflect")
    crops = padded.unfold(2, images.shape[2], 1).unfold(
        3, images.shape[3], 1
    )
    batch_indices = torch.arange(images.shape[0], device=images.device)
    offsets_y = torch.randint(
        0, 2 * padding + 1, (images.shape[0],), device=images.device
    )
    offsets_x = torch.randint(
        0, 2 * padding + 1, (images.shape[0],), device=images.device
    )
    images = crops[batch_indices, :, offsets_y, offsets_x]
    return images, labels
>>>>>>> REPLACE