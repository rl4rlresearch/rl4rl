MECHANISM: Balanced integer-translation augmentation

HYPOTHESIS: Adding a uniformly cycled ±2-pixel translation to flip augmentation will exceed 9,240 correct predictions by improving positional robustness without increasing parameters or model-forward cost.

INTENDED_EDIT: Apply all 25 integer translations within a two-pixel radius evenly across training steps, using replicated padding while retaining per-image random horizontal flips.

EVIDENCE: The 12.5%-floor model reached 9,240 correct, while further schedule refinement and classifier widening failed to improve it; translation augmentation is an orthogonal, computationally inexpensive change to the strongest verified design.

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
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    translation = step % 25
    offset_y = translation // 5
    offset_x = translation % 5
    images = F.pad(images, (2, 2, 2, 2), mode="replicate")
    images = images[:, :, offset_y : offset_y + 28, offset_x : offset_x + 28]
    return images, labels
>>>>>>> REPLACE