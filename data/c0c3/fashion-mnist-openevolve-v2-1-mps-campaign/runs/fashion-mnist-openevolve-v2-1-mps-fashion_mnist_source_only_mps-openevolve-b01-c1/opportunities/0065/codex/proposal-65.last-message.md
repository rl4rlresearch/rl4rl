MECHANISM: Vectorized mild rotation augmentation

HYPOTHESIS: Adding independent ±8° rotations before the proven translation-and-flip pipeline will exceed 9,247 correct predictions by learning a missing orientation invariance without changing model capacity or view supervision.

INTENDED_EDIT: Apply a vectorized random rotation to each training image before reflected padding and crop construction.

EVIDENCE: Schedule refinements plateaued at or below 9,247 correct, while additional translation microbatching regressed to 9,237; this motivates testing an orthogonal geometric augmentation while preserving the successful crop ensemble and loss curriculum.

<<<<<<< SEARCH
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    offsets = torch.randint(0, 5, (2,))
=======
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    max_angle = 8.0 * math.pi / 180.0
    angles = (
        2.0
        * torch.rand(
            images.shape[0],
            device=images.device,
            dtype=images.dtype,
        )
        - 1.0
    ) * max_angle
    cosines = angles.cos()
    sines = angles.sin()
    theta = torch.zeros(
        (images.shape[0], 2, 3),
        device=images.device,
        dtype=images.dtype,
    )
    theta[:, 0, 0] = cosines
    theta[:, 0, 1] = -sines
    theta[:, 1, 0] = sines
    theta[:, 1, 1] = cosines
    rotation_grid = F.affine_grid(
        theta,
        images.shape,
        align_corners=False,
    )
    images = F.grid_sample(
        images,
        rotation_grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
    offsets = torch.randint(0, 5, (2,))
>>>>>>> REPLACE