MECHANISM: Annealed random-erasing augmentation

HYPOTHESIS: Replacing the symmetry-redundant random flips with mild 6×6 random erasing will exceed 9,328 correct predictions by adding useful occlusion robustness, while annealing erasing during EMA collection preserves clean-image convergence.

INTENDED_EDIT: Apply per-image 6×6 mean-filled erasing with probability 0.5 through the first half of training, then linearly decay its probability to zero.

EVIDENCE: The classifier is already exactly invariant to horizontal input flips because every prediction symmetrically fuses both orientations, so the current flip augmentation supplies no new training views; meanwhile, repeated classifier, pooling, dilation, and late-loss changes failed to improve 9,328 correct.

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
    progress = min(step / max(total_steps, 1), 1.0)
    erase_decay = max(progress - 0.50, 0.0) / 0.50
    erase_probability = 0.50 * (1.0 - erase_decay)

    batch_size, _, height, width = images.shape
    erase_size = 6
    erase_sample = (
        torch.rand(batch_size, device=images.device) < erase_probability
    )
    erase_top = torch.randint(
        height - erase_size + 1,
        (batch_size,),
        device=images.device,
    )
    erase_left = torch.randint(
        width - erase_size + 1,
        (batch_size,),
        device=images.device,
    )
    rows = torch.arange(height, device=images.device)[None, :, None]
    columns = torch.arange(width, device=images.device)[None, None, :]
    erase_mask = (
        erase_sample[:, None, None]
        & (rows >= erase_top[:, None, None])
        & (rows < erase_top[:, None, None] + erase_size)
        & (columns >= erase_left[:, None, None])
        & (columns < erase_left[:, None, None] + erase_size)
    )
    images = images.masked_fill(erase_mask[:, None], 0.0)
    return images, labels
>>>>>>> REPLACE