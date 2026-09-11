MECHANISM: Per-image replicated-border translation augmentation

HYPOTHESIS: Adding independent two-pixel translations to the fast 9,320-correct hard-maximum model will exceed 9,322 correct predictions by learning small positional invariance without materially increasing training time.

INTENDED_EDIT: Replace the redundant preparation-time horizontal flip with per-image random crops from a two-pixel replicated border; paired-view loss still supplies both horizontal orientations.

EVIDENCE: Hard-maximum attention reached 9,320 correct in 75.3 seconds, while top-four gained only two predictions at 78.8 seconds and later attention variants timed out; the paired loss already evaluates both horizontal orientations, making preparation-time flipping available for a low-overhead orthogonal augmentation.

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
    padding = 2
    batch_size, channels, height, width = images.shape
    padded_images = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    ).squeeze(1)

    row_starts = torch.randint(
        0,
        2 * padding + 1,
        (batch_size, 1, 1),
        device=images.device,
    )
    column_starts = torch.randint(
        0,
        2 * padding + 1,
        (batch_size, 1, 1),
        device=images.device,
    )
    batch_indices = torch.arange(
        batch_size,
        device=images.device,
    )[:, None, None]
    row_indices = row_starts + torch.arange(
        height,
        device=images.device,
    )[None, :, None]
    column_indices = column_starts + torch.arange(
        width,
        device=images.device,
    )[None, None, :]
    images = padded_images[
        batch_indices,
        row_indices,
        column_indices,
    ].unsqueeze(1)
    return images, labels
>>>>>>> REPLACE