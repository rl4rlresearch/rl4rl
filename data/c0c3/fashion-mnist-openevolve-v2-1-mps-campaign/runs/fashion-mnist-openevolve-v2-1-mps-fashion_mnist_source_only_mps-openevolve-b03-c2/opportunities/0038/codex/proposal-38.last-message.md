MECHANISM: Compute-free near-zero cosine stabilization

HYPOTHESIS: Reference Design 3’s dense batch-32 matched-view training with a 2% terminal learning-rate floor will exceed 9,214 correct predictions by reducing late-update noise without adding runtime overhead.

INTENDED_EDIT: Restore the strongest verified batch-32 flip-paired procedure, retain efficient pair-batched ten-view inference, and lower the final learning rate from 2e-4 to 4e-5.

EVIDENCE: Reference Design 3 achieved 9,214 correct, while 0.995 EMA reached 9,241 before timing out; this motivates targeting the same late-training instability through a zero-overhead lower cosine floor.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 32
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=2.5e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
=======
        lr=2.0e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
>>>>>>> REPLACE

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

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    del step, total_steps
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    multiplier = 0.02 + 0.98 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE