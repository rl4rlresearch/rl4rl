MECHANISM: Compute-neutral inference-orbit translation curriculum

HYPOTHESIS: Replacing ±2-pixel and diagonal training crops with the exact five spatial views used by ten-view inference will exceed 9,208 correct predictions without increasing training time.

INTENDED_EDIT: Cycle batch-wide through center and four one-pixel cardinal translations during the augmented half, while removing redundant random flips because the loss already supervises both orientations.

EVIDENCE: Flip-paired supervision improved accuracy from 9,142 to 9,208, while more expensive translation-paired variants timed out; matching the inference translations through the existing single-crop path tests the same alignment principle at lower cost.

<<<<<<< SEARCH
    if step < total_steps // 2:
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        top = step % 5
        left = (step // 5) % 5
        images = padded[:, :, top : top + 28, left : left + 28]
        flip_mask = torch.rand(
            (images.shape[0], 1, 1, 1), device=images.device
        ) < 0.5
        images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
=======
    if step < total_steps // 2:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        view = step % 5
        top = (1, 0, 2, 1, 1)[view]
        left = (1, 1, 1, 0, 2)[view]
        images = padded[:, :, top : top + 28, left : left + 28]
    return images, labels
>>>>>>> REPLACE