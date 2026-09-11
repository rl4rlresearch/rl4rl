MECHANISM: Intermediate single-microbatch update frequency

HYPOTHESIS: A 196,608-token optimizer batch will retain at least 485M training tokens while reducing val_bpb below 0.985237.

INTENDED_EDIT: Use a 96-sequence device batch and optimizer batch three-quarters the current size, preserving one microbatch per update and all learning rates and schedules.

EVIDENCE: The 262K-token batch achieved 0.985237, while neighboring 524K and 131K batches regressed to 0.991682 and 0.989256; testing 197K refines the clearly bracketed update-frequency optimum.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 96   # one 197K-token microbatch per optimizer step
>>>>>>> REPLACE