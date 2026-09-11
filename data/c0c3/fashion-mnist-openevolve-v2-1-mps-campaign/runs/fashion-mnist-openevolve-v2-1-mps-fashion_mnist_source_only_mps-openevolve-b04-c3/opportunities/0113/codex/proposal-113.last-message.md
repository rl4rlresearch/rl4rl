MECHANISM: Compute-light cyclic translation augmentation

HYPOTHESIS: Deterministically cycling replicate-padded ±1-pixel translations during training will increase validation_correct above 9,243 without materially increasing training time.

INTENDED_EDIT: Replace per-batch unshifted training with a nine-position translation cycle while retaining random horizontal flips and the best verified inference calibration.

EVIDENCE: Inference temperature and power-mean refinements plateaued at exactly 9,243 correct; the prior per-example translation attempt timed out, motivating a cheaper batch-wide pad-and-slice implementation that adds no model forwards.

<<<<<<< SEARCH
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    return images, labels
=======
    del total_steps
    height, width = images.shape[-2:]
    offset_x = step % 3
    offset_y = (step // 3) % 3
    images = F.pad(images, (1, 1, 1, 1), mode="replicate")
    images = images[
        ...,
        offset_y : offset_y + height,
        offset_x : offset_x + width,
    ]
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    return images, labels
>>>>>>> REPLACE