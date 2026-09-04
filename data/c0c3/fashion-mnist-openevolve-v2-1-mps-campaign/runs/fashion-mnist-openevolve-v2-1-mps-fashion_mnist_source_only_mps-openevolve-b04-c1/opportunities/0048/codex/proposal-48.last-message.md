MECHANISM: Per-example translation augmentation

HYPOTHESIS: Training on independently translated crops of up to two pixels will exceed 9,316 correct predictions by improving positional robustness without altering the successful mirrored-feature fusion.

INTENDED_EDIT: Replace the functionally redundant random horizontal flip with efficient replicate-padded, per-image random translation.

EVIDENCE: Joint invariant–disagreement fusion reached 9,316 correct, while added fusion capacity and altered annealing failed to improve it. Because the current fusion is unchanged when its two mirrored views are swapped, the existing random flip adds no training variation; translation supplies a distinct spatial perturbation without parameters or extra model passes.

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
    height, width = images.shape[-2:]
    channels = images.shape[1]
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    offsets = torch.randint(
        -padding,
        padding + 1,
        (images.shape[0], 2),
        device=images.device,
    )
    row_indices = (
        torch.arange(height, device=images.device)[None, :]
        + offsets[:, 0, None]
        + padding
    )
    images = torch.gather(
        padded,
        2,
        row_indices[:, None, :, None].expand(
            -1, channels, -1, padded.shape[-1]
        ),
    )
    column_indices = (
        torch.arange(width, device=images.device)[None, :]
        + offsets[:, 1, None]
        + padding
    )
    images = torch.gather(
        images,
        3,
        column_indices[:, None, None, :].expand(
            -1, channels, height, -1
        ),
    )
    return images, labels
>>>>>>> REPLACE