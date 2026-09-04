MECHANISM: Two-microbatch 393K-token updates

HYPOTHESIS: Using 96-sequence microbatches with two-way accumulation will retain more of the baseline’s throughput while increasing update frequency; this will beat the baseline val_bpb of 0.995558.

INTENDED_EDIT: Reduce the global batch to 393K tokens and device batch to 96, preserving two microbatches per optimizer step.

EVIDENCE: The 192-sequence single-microbatch design fell to 416.8M tokens and 33.24% MFU versus the baseline’s 497.0M tokens and 39.58% MFU, so its 1.000243 val_bpb does not isolate the benefit of smaller, more frequent updates.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**17 # ~393K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 96   # two microbatches per optimizer step
>>>>>>> REPLACE