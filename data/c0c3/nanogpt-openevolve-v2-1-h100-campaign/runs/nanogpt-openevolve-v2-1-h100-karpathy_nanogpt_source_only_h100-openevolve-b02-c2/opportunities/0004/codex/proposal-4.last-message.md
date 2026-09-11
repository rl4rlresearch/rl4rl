MECHANISM: Quarter-megabyte-batch update cadence

HYPOTHESIS: Halving the optimizer batch to 131K tokens will lower val_bpb below 0.987182 by doubling update opportunities, despite the smaller device batch’s potential throughput cost.

INTENDED_EDIT: Reduce both total and device batch sizes by half, retaining one forward/backward pass per optimizer step and all other architecture and optimization settings.

EVIDENCE: Halving the prior 524K-token batch to 262K preserved nearly all token throughput (489.7M versus 497.0M) while improving val_bpb from 0.995558 to 0.987182, directly motivating another batch-size reduction.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step; one device batch
=======
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step; one device batch
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 64   # per-device batch size; one 131K-token optimizer batch
>>>>>>> REPLACE