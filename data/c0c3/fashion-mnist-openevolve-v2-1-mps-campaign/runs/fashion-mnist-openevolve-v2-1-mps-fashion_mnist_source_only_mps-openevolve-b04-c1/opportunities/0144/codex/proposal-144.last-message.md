MECHANISM: Training-only integer translation augmentation

HYPOTHESIS: Replacing the symmetry-null random flip with uniformly sampled ±2-pixel translations will exceed 9,328 correct predictions by teaching local shift robustness while preserving the verified architecture, loss, and inference path.

INTENDED_EDIT: Replace random horizontal flipping with vectorized per-image crops from replicate-padded inputs.

EVIDENCE: The 9,328 baseline’s paired invariant/disagreement fusion makes its current random flip augmentation functionally redundant, while inference-fusion changes and auxiliary supervision regressed; this isolates a training-only source of genuinely new examples.

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
    max_shift = 2
    padded = F.pad(
        images,
        (max_shift, max_shift, max_shift, max_shift),
        mode="replicate",
    )
    offsets = torch.randint(
        0,
        2 * max_shift + 1,
        (images.shape[0], 2),
        device=images.device,
    )
    rows = torch.arange(images.shape[-2], device=images.device)
    columns = torch.arange(images.shape[-1], device=images.device)
    crop_indices = (
        (rows[None, :, None] + offsets[:, 0, None, None])
        * padded.shape[-1]
        + columns[None, None, :]
        + offsets[:, 1, None, None]
    )
    images = torch.gather(
        padded.flatten(2),
        2,
        crop_indices.flatten(1)[:, None, :].expand(
            -1, images.shape[1], -1
        ),
    ).reshape_as(images)
    return images, labels
>>>>>>> REPLACE