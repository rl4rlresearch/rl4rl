MECHANISM: Runtime-safe intermediate-batch matched-view optimization

HYPOTHESIS: Batch 48 with 2,084 dense updates and exact cardinal-view augmentation will finish within the runtime limit and exceed the current 9,119 correct predictions.

INTENDED_EDIT: Reduce batch size from 64 to 48, interpolate the learning rate to 2.25e-3, and replace the broader ±2-pixel crop grid with the five views used at inference.

EVIDENCE: Reference Design 3 reached 9,214 correct using batch-32 dense optimization and matched cardinal views, while the current batch-64 design finishes in 63.63 seconds; batch 48 captures additional updates and view matching without the timeout-prone 3,126-step regime.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0
=======
BATCH_SIZE = 48
GRAD_CLIP_NORM = 5.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=2.5e-3,
        betas=(0.9, 0.99),
=======
        lr=2.25e-3,
        betas=(0.9, 0.99),
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
        group["lr"] = 2.5e-3 * multiplier
=======
        group["lr"] = 2.25e-3 * multiplier
>>>>>>> REPLACE