MECHANISM: Inverse one-cycle AdamW momentum

HYPOTHESIS: Cycling AdamW’s first-moment coefficient from 0.95 to 0.85 and back to 0.95 will exceed 9,262 correct predictions by improving adaptation near peak learning rate and smoothing convergence late in training without adding optimizer steps or compute-heavy model capacity.

INTENDED_EDIT: Preserve the best architecture, augmentation, and 0.020→0.009 smoothing schedule while inversely coupling AdamW beta1 to the existing one-cycle learning-rate schedule.

EVIDENCE: The 233,434-parameter design with a 0.009 smoothing floor achieved 9,262 correct, while head widening and feature-channel reallocation regressed and the smaller-batch optimization test timed out; a zero-overhead change to the otherwise fixed optimizer trajectory is therefore the most informative next probe.

<<<<<<< SEARCH
        betas=(0.9, 0.99),
=======
        betas=(0.95, 0.99),
>>>>>>> REPLACE

<<<<<<< SEARCH
    if progress < warmup_fraction:
        multiplier = 0.1 + 0.9 * progress / warmup_fraction
    else:
        cosine_progress = (
            progress - warmup_fraction
        ) / (1.0 - warmup_fraction)
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
=======
    if progress < warmup_fraction:
        warmup_progress = progress / warmup_fraction
        multiplier = 0.1 + 0.9 * warmup_progress
        beta1 = 0.95 - 0.10 * warmup_progress
    else:
        cosine_progress = (
            progress - warmup_fraction
        ) / (1.0 - warmup_fraction)
        cosine_value = math.cos(math.pi * cosine_progress)
        multiplier = 0.03 + 0.97 * 0.5 * (1.0 + cosine_value)
        beta1 = 0.85 + 0.10 * 0.5 * (1.0 - cosine_value)
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
        group["betas"] = (beta1, 0.99)
>>>>>>> REPLACE