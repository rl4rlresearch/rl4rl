MECHANISM: Per-example translation augmentation

HYPOTHESIS: Replacing the symmetry-redundant random flip with random two-pixel translations will exceed 9,328 correct predictions by improving positional robustness without changing model size or materially increasing convolutional compute.

INTENDED_EDIT: Pad each training image by two pixels using replicated borders, then independently sample and extract a translated 28×28 crop for every example.

EVIDENCE: The current mirrored-view fusion already makes horizontal flips output-equivalent and achieved 9,328 correct; multiple subsequent regularization-timing changes failed to improve it, motivating an orthogonal augmentation that introduces genuinely new views.

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
    padding = 2
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    offsets_y = torch.randint(
        0,
        2 * padding + 1,
        (batch_size, 1),
        device=images.device,
    )
    offsets_x = torch.randint(
        0,
        2 * padding + 1,
        (batch_size, 1),
        device=images.device,
    )
    rows = offsets_y + torch.arange(height, device=images.device)
    cols = offsets_x + torch.arange(width, device=images.device)
    images = padded.gather(
        2,
        rows[:, None, :, None].expand(
            batch_size,
            channels,
            height,
            width + 2 * padding,
        ),
    )
    images = images.gather(
        3,
        cols[:, None, None, :].expand(
            batch_size,
            channels,
            height,
            width,
        ),
    )
    return images, labels
>>>>>>> REPLACE