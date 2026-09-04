MECHANISM: Alternating one-pixel translation augmentation

HYPOTHESIS: Training on original images for half the steps and deterministic one-pixel translations for the other half will exceed 9,251 correct predictions by improving local shift robustness without materially increasing runtime.

INTENDED_EDIT: Extend the existing horizontal-flip augmentation with balanced, deterministic one-pixel horizontal, vertical, and diagonal translations on alternating optimizer steps.

EVIDENCE: The best completed design reached 9,251 correct using horizontal flips alone, while repeated inference-blend refinements failed to yield new measurements and a deeper architecture timed out; this tests an unexplored training-side invariance with negligible parameter and compute cost.

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

    if step % 2:
        translations = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
        shift_y, shift_x = translations[(step // 2) % len(translations)]
        images = F.pad(images, (1, 1, 1, 1), mode="replicate")
        images = images[
            :,
            :,
            1 + shift_y : 29 + shift_y,
            1 + shift_x : 29 + shift_x,
        ]
    return images, labels
>>>>>>> REPLACE