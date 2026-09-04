MECHANISM: Low-overhead cyclic one-pixel translation augmentation

HYPOTHESIS: Cycling through all nine ±1-pixel translations will exceed 9,240 correct predictions by improving positional robustness while avoiding the stronger perturbation and padding overhead of the timed-out ±2-pixel design.

INTENDED_EDIT: Add a batch-shared, evenly cycled one-pixel translation via `torch.roll` before the existing random horizontal flip, preserving the strongest verified architecture and 12.5%-floor schedule.

EVIDENCE: The current design achieved the best verified result at 9,240 correct, while the orthogonal ±2-pixel replicated-padding augmentation timed out; a smaller roll-based variant retains the promising invariance mechanism with less augmentation severity and overhead.

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
    shift_x = step % 3 - 1
    shift_y = (step // 3) % 3 - 1
    images = torch.roll(images, shifts=(shift_y, shift_x), dims=(-2, -1))
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    return images, labels
>>>>>>> REPLACE