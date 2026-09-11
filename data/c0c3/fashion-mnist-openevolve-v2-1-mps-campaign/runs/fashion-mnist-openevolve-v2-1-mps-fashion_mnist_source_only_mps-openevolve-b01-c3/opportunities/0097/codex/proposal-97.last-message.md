MECHANISM: Integer-bucketed train–evaluation view-prior alignment

HYPOTHESIS: Sampling training crops with the evaluation pool’s exact 1.546875× center prior will exceed 9,287 correct predictions by emphasizing validation-aligned examples while retaining substantial shifted-view augmentation.

INTENDED_EDIT: Replace uniform crop-position sampling with a low-overhead 99:64:64:64:64 integer-bucket sampler; preserve the best verified evaluation pooling and calibration.

EVIDENCE: The 1.546875× center-biased evaluation pool retained 9,287 correct and improved cross-entropy; the prior training-alignment attempt timed out, so its accuracy effect remains untested, and integer bucketing avoids multinomial-sampling overhead.

<<<<<<< SEARCH
    positions = torch.randint(0, 5, (batch,), device=images.device)
=======
    position_draws = torch.randint(0, 355, (batch,), device=images.device)
    positions = torch.where(
        position_draws < 99,
        0,
        1 + (position_draws - 99) // 64,
    )
>>>>>>> REPLACE