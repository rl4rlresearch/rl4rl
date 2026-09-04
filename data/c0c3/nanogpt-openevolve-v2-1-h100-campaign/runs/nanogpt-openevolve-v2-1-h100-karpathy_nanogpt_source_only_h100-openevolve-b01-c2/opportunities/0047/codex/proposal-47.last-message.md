MECHANISM: Equal-area cosine warmdown at the best batch operating point

HYPOTHESIS: Using cosine rather than linear warmdown with the proven 258,048-token batch and `MATRIX_LR=0.040` will retain roughly 520M training tokens while lowering `val_bpb` below 0.985181.

INTENDED_EDIT: Restore the best 126-sequence microbatch and replace the 60% linear warmdown with an equal-duration, equal-area cosine warmdown.

EVIDENCE: The 258,048-token, `MATRIX_LR=0.040` design achieved the best `val_bpb` of 0.985181, while nearby batch sizes and matrix learning rates regressed; cosine warmdown tests annealing curvature without changing the proven peak LR, endpoints, or total LR exposure.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 15 * 2**14 # ~246K tokens; one 120-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 63 * 2**12 # ~258K tokens; one 126-sequence microbatch per step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 120  # one microbatch per 245,760-token optimizer step
=======
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 126  # one microbatch per 258,048-token optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine = 0.5 * (1.0 - math.cos(math.pi * cooldown))
        return cosine + (1 - cosine) * FINAL_LR_FRAC
>>>>>>> REPLACE