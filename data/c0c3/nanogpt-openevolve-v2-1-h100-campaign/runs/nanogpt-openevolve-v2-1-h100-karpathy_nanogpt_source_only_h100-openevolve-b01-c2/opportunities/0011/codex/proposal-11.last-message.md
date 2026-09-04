MECHANISM: Cosine learning-rate warmdown on the proven high-throughput configuration

HYPOTHESIS: Restoring the 128-sequence, 524K-token SSSL baseline and replacing its final-half linear decay with an equal-area cosine decay will retain roughly 497M-token throughput while allowing smaller late-training updates, lowering val_bpb below 0.995558.

INTENDED_EDIT: Restore Reference Design 1’s batching and change only the shape of the final-half learning-rate warmdown from linear to cosine.

EVIDENCE: Reference Design 1 achieved the best val_bpb, 0.995558, at 497.0M tokens; every tested batching change reduced throughput or worsened validation, while delaying warmdown also worsened validation, motivating a schedule-shape test that preserves the proven batch and cooldown duration.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**17 # 393K tokens; one 192-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 192  # use available H100 memory to eliminate accumulation
=======
DEVICE_BATCH_SIZE = 128  # proven high-throughput per-device batch size
>>>>>>> REPLACE

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown_progress = (progress - (1.0 - WARMDOWN_RATIO)) / WARMDOWN_RATIO
        cosine = 0.5 * (1.0 + math.cos(math.pi * cooldown_progress))
        return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * cosine
>>>>>>> REPLACE