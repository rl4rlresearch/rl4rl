MECHANISM: Higher-frequency single-microbatch optimization

HYPOTHESIS: Halving the optimizer batch again to 131,072 tokens will reduce val_bpb below 0.986162 by providing roughly twice as many parameter updates, despite a possible throughput reduction from the smaller device batch.

INTENDED_EDIT: Reduce both total and device batch sizes by half, preserving one forward/backward pass per optimizer step and all other model and optimizer settings.

EVIDENCE: Halving the optimizer batch from 524,288 to 262,144 tokens improved val_bpb from 0.993365 to 0.986162 even though trained tokens fell from 512.2M to 498.1M, strongly implicating update frequency rather than throughput as the gain.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 64   # per-device batch size; one microbatch per optimizer step
>>>>>>> REPLACE