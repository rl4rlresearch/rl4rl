MECHANISM: Mild stochastic rotation augmentation

HYPOTHESIS: Applying per-image rotations up to ±7.5° on half of training examples will exceed 9,267 correct predictions by learning a missing pose invariance without altering the successful architecture or validation ensemble.

INTENDED_EDIT: Add probability-0.5 bilinear random rotation in `prepare_training_batch`, using reflection padding and preserving the existing batch interface.

EVIDENCE: Matching training to the existing validation crop distribution reduced accuracy to 9,247, and inference-aligned supervision reduced it to 9,237; this motivates testing a distinct, semantically safe augmentation axis rather than further refining the exhausted translation/ensemble formulation.

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    batch_size = images.shape[0]
    angles = (
        torch.rand(batch_size, device=images.device) - 0.5
    ) * math.radians(15.0)
    angles = angles * (
        torch.rand(batch_size, device=images.device) < 0.5
    ).to(images.dtype)
    cosines = angles.cos()
    sines = angles.sin()
    theta = images.new_zeros((batch_size, 2, 3))
    theta[:, 0, 0] = cosines
    theta[:, 0, 1] = -sines
    theta[:, 1, 0] = sines
    theta[:, 1, 1] = cosines
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    images = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
    return images, labels
>>>>>>> REPLACE