MECHANISM: Conservative per-example translation augmentation

HYPOTHESIS: Replacing the redundant pre-pairing flip with ±2-pixel translations on half the examples will exceed 9,322 correct predictions by improving spatial robustness while retaining centered training views and the strongest top-four attention design.

INTENDED_EDIT: Apply efficient per-example random crops from replicate-padded images; explicit paired-view training continues to provide both horizontal orientations.

EVIDENCE: Top-four attention is the best verified architecture at 9,322 correct, while numerous attention and loss changes regressed or timed out. The current preparation flip adds no distinct orientation because training_loss already includes every image and its flip, motivating an orthogonal, low-overhead augmentation.

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
    translation_mask = (
        torch.rand(batch_size, 1, device=images.device) < 0.5
    )
    shifts = torch.randint(
        -2, 3, (batch_size, 2), device=images.device
    )
    shifts = shifts * translation_mask

    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    rows = (
        torch.arange(height, device=images.device)[None, :, None]
        + shifts[:, 0, None, None]
        + 2
    )
    columns = (
        torch.arange(width, device=images.device)[None, None, :]
        + shifts[:, 1, None, None]
        + 2
    )
    crop_indices = rows * (width + 4) + columns
    crop_indices = crop_indices.flatten(1).unsqueeze(1)
    crop_indices = crop_indices.expand(-1, channels, -1)
    images = padded.flatten(2).gather(2, crop_indices).reshape_as(images)
    return images, labels
>>>>>>> REPLACE