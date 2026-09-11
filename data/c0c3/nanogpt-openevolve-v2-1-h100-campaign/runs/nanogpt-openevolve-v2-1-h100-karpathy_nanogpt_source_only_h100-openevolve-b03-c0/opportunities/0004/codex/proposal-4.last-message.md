MECHANISM: Intermediate update-density interpolation

HYPOTHESIS: A 96-sequence single-microbatch step will preserve near-128-batch throughput while adding 33% more optimizer updates, achieving `val_bpb < 0.987111`.

INTENDED_EDIT: Reduce both the global token batch and device batch from 128 to 96 sequences, retaining one microbatch per optimizer step.

EVIDENCE: Batch 128 achieved the best `val_bpb` of 0.987111 at 489.9M tokens, while batch 64 regressed to 0.991095 despite more updates because throughput fell to 476.7M tokens; batch 96 tests the implied intermediate optimum.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step; one efficient 128-sequence microbatch
=======
TOTAL_BATCH_SIZE = 96 * 2048 # ~197K tokens per optimizer step; one 96-sequence microbatch
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 96   # per-device batch size; one microbatch per optimizer step
>>>>>>> REPLACE