MECHANISM: Mild random-erasing regularization

HYPOTHESIS: Applying a 4×4 mean-valued erasure to 25% of training images will exceed 9,290 correct predictions by improving local-occlusion robustness without disturbing the validated translation, flip, smoothing, or EMA settings.

INTENDED_EDIT: Add inexpensive per-image random erasing after the existing cardinal-shift and horizontal-flip augmentation.

EVIDENCE: Removing centered training views reduced validation correct from 9,290 to 9,270, while dropout and label-smoothing changes also regressed; this motivates preserving those validated components and testing an orthogonal, low-cost augmentation.

<<<<<<< SEARCH
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    return images, labels
=======
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )

    erase_size = 4
    erase_mask = torch.rand(images.shape[0], device=images.device) < 0.25
    erase_y = torch.randint(
        0, 29 - erase_size, (images.shape[0],), device=images.device
    )
    erase_x = torch.randint(
        0, 29 - erase_size, (images.shape[0],), device=images.device
    )
    rows = torch.arange(28, device=images.device)[None, :, None]
    columns = torch.arange(28, device=images.device)[None, None, :]
    cutouts = (
        (rows >= erase_y[:, None, None])
        & (rows < erase_y[:, None, None] + erase_size)
        & (columns >= erase_x[:, None, None])
        & (columns < erase_x[:, None, None] + erase_size)
        & erase_mask[:, None, None]
    )
    images = images.masked_fill(cutouts[:, None], 0.0)
    return images, labels
>>>>>>> REPLACE