MECHANISM: Cosine warmdown on the proven 256-token local-attention baseline

HYPOTHESIS: Restoring seven 256-token local layers and using cosine rather than linear warmdown will retain roughly 525M tokens while lowering val_bpb below 0.992854.

INTENDED_EDIT: Restore the best-performing 256-token windows and replace the linear 50%-window LR decay with an equal-duration cosine decay.

EVIDENCE: The 256-token design achieved the best observed val_bpb, 0.992854; nearby uniform windows and progressive allocation did not improve it, motivating an orthogonal schedule refinement without changing its architecture or batch.

<<<<<<< SEARCH
        short_window = 3 * long_window // 32
=======
        short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # seven 192-token local layers, then one forced full-context anchor
=======
WINDOW_PATTERN = "SSSS" # seven 256-token local layers, then one forced full-context anchor
>>>>>>> REPLACE

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine_decay = 0.5 - 0.5 * math.cos(math.pi * cooldown)
        return cosine_decay + (1 - cosine_decay) * FINAL_LR_FRAC
>>>>>>> REPLACE