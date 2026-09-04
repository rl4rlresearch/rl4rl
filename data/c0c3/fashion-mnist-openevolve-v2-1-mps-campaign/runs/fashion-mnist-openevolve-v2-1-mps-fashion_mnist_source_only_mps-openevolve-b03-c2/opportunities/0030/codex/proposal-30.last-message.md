MECHANISM: Compute-capped late flip-pairing curriculum

HYPOTHESIS: Batch-32 dense optimization with matched cardinal augmentation and flip-paired supervision restricted to the final quarter will finish within the time limit and exceed 9,214 correct predictions.

INTENDED_EDIT: Use Reference Design 3’s batch size, learning rate, and five inference-matched spatial views, while limiting duplicated flip forwards to the final 25% of updates.

EVIDENCE: Reference Design 3 completed with 9,214 correct using batch-32 flip pairing throughout, while a single-view-first, paired-second-half variant timed out; quarter-length late pairing reduces training forwards from 150,000 to 125,000 while retaining late exact symmetry.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 32
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=2.5e-3,
=======
        lr=2.0e-3,
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
=======
    if step < total_steps // 2:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        view = step % 5
        top = (1, 0, 2, 1, 1)[view]
        left = (1, 1, 1, 0, 2)[view]
        images = padded[:, :, top : top + 28, left : left + 28]
        flip_mask = torch.rand(
            (images.shape[0], 1, 1, 1), device=images.device
        ) < 0.5
        images = torch.where(flip_mask, images.flip(-1), images)
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    if step >= (3 * total_steps) // 4:
        images = torch.cat((images, images.flip(-1)), dim=0)
        labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.5e-3 * multiplier
=======
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE