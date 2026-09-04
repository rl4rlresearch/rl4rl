MECHANISM: Mild per-image translation augmentation

HYPOTHESIS: Applying one-pixel translations to roughly half of training images will exceed 9,258 correct predictions by improving local shift invariance without changing the model, optimizer-step count, or evaluation path.

INTENDED_EDIT: Add vectorized, training-only random crops from replicate-padded images, retaining the original position for half the batch.

EVIDENCE: The current batch-32 EMA design achieved 9,258 correct, while 0.05 label smoothing outperformed both hard targets and stronger 0.10 smoothing; this motivates another mild, parameter-free regularizer rather than a larger architectural or computational change.

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    batch_size = images.shape[0]
    offsets = torch.randint(0, 3, (batch_size, 2), device=images.device)
    keep_centered = torch.rand(batch_size, device=images.device) < 0.5
    offsets = torch.where(
        keep_centered[:, None], torch.ones_like(offsets), offsets
    )

    height, width = images.shape[-2:]
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    crops = padded.unfold(2, height, 1).unfold(3, width, 1)
    batch_indices = torch.arange(batch_size, device=images.device)
    images = crops[
        batch_indices, :, offsets[:, 0], offsets[:, 1]
    ]
    return images, labels
>>>>>>> REPLACE