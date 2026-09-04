MECHANISM: Mild random translation augmentation

HYPOTHESIS: Replacing the redundant random flip with one-pixel translations will exceed 9,330 correct predictions by improving spatial robustness while retaining exact flip invariance and the proven architecture.

INTENDED_EDIT: Apply a random one-pixel horizontal and vertical translation to half of training examples using replicate padding; remove the output-equivalent random flip.

EVIDENCE: Multiple capacity and spatial-refinement changes underperformed, while the current symmetric fusion already makes horizontal flipping prediction-invariant, so the existing augmentation adds no meaningful diversity.

<<<<<<< SEARCH
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
    return images, labels
=======
    del step, total_steps
    batch_size, channels, height, width = images.shape
    apply_shift = (
        torch.rand(batch_size, device=images.device) < 0.5
    )
    offset_y = torch.randint(
        0, 3, (batch_size,), device=images.device
    )
    offset_x = torch.randint(
        0, 3, (batch_size,), device=images.device
    )
    centered = torch.ones_like(offset_y)
    offset_y = torch.where(apply_shift, offset_y, centered)
    offset_x = torch.where(apply_shift, offset_x, centered)

    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    rows = (
        torch.arange(height, device=images.device)[None, :]
        + offset_y[:, None]
    )
    shifted_rows = padded.gather(
        2,
        rows[:, None, :, None].expand(
            -1, channels, -1, padded.shape[-1]
        ),
    )
    columns = (
        torch.arange(width, device=images.device)[None, :]
        + offset_x[:, None]
    )
    images = shifted_rows.gather(
        3,
        columns[:, None, None, :].expand(
            -1, channels, height, -1
        ),
    )
    return images, labels
>>>>>>> REPLACE