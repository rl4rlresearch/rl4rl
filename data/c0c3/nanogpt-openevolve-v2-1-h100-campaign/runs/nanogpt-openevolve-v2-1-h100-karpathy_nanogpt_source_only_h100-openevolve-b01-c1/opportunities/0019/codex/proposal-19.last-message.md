MECHANISM: 131K-token high-frequency optimization

HYPOTHESIS: Halving the optimizer batch again to 131K tokens will increase parameter-update frequency enough to lower val_bpb below 0.986167, despite some throughput loss from the smaller device batch and additional optimizer work.

INTENDED_EDIT: Reduce both TOTAL_BATCH_SIZE and DEVICE_BATCH_SIZE by half, retaining one microbatch per optimizer step and leaving the successful 50.3M-parameter architecture, learning rates, and final full-context layer unchanged.

EVIDENCE: Halving the batch from 524K to 262K tokens nearly doubled update frequency at unchanged token throughput and improved val_bpb from 0.992286 to 0.986167, the strongest verified gain; continuing the same controlled batch-size sweep tests whether optimization cadence remains the limiting factor.

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