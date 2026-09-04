MECHANISM: Balanced one-pixel translation augmentation

HYPOTHESIS: Cycling through all nine one-pixel translations will exceed 9,322 correct predictions by improving spatial robustness while retaining the reliable hard-maximum architecture and runtime.

INTENDED_EDIT: Apply a deterministic, balanced 3×3 translation schedule after the existing random flip, using replicated border padding and no additional RNG draws.

EVIDENCE: The reliable hard-maximum model repeatedly reaches 9,320 correct, while attention, consistency, schedule, and EMA changes either regress or risk timeout; mild translation augmentation is an orthogonal, low-cost source of invariance not tested by those results.

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
    del total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )

    shift_code = step % 9
    vertical_offset = shift_code // 3
    horizontal_offset = shift_code % 3
    images = F.pad(images, (1, 1, 1, 1), mode="replicate")
    images = images[
        :,
        :,
        vertical_offset : vertical_offset + 28,
        horizontal_offset : horizontal_offset + 28,
    ]
    return images, labels
>>>>>>> REPLACE