MECHANISM: Quarter-sized effective batch with single-microbatch updates

HYPOTHESIS: Halving the effective batch again will improve val_bpb below 0.987174 by roughly doubling optimizer updates, while retaining enough of the 489.9M-token throughput to offset the increased gradient noise and optimizer overhead.

INTENDED_EDIT: Reduce both total and device batch sizes from 262K/128 to 131K/64 tokens/sequences, preserving one microbatch per optimizer step.

EVIDENCE: The previous batch halving nearly preserved throughput (489.9M versus 497.0M tokens), increased updates from 948 to 1869, and improved val_bpb from 0.995558 to 0.987174, making a second halving the clearest test of whether update frequency remains the limiting factor.

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