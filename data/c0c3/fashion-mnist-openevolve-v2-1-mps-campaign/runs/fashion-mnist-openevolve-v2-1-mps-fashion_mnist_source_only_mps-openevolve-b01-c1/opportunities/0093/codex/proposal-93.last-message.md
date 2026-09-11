MECHANISM: Batch-shared mild rotational augmentation

HYPOTHESIS: Applying one shared random rotation of at most 8° per training batch will exceed 9,247 correct predictions by learning modest orientation invariance while preserving the batch-correlated augmentation structure that outperformed per-example transforms.

INTENDED_EDIT: Rotate each training batch by a uniformly sampled angle in [−8°, 8°] with bilinear sampling and reflected boundaries before generating the existing translated and flipped views.

EVIDENCE: Per-example translation sampling regressed to 9,219 correct while batch-shared offsets remained stronger; recent capacity, EMA, and target-regularization changes also regressed, motivating a new invariance applied with the proven batch-shared structure.

<<<<<<< SEARCH
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    offsets = torch.randint(0, 5, (2,))
=======
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    angle = (
        2.0 * torch.rand((), device=images.device) - 1.0
    ) * math.radians(8.0)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    transform = images.new_zeros((images.shape[0], 2, 3))
    transform[:, 0, 0] = cosine
    transform[:, 0, 1] = -sine
    transform[:, 1, 0] = sine
    transform[:, 1, 1] = cosine
    grid = F.affine_grid(
        transform,
        images.shape,
        align_corners=False,
    )
    images = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
    offsets = torch.randint(0, 5, (2,))
>>>>>>> REPLACE