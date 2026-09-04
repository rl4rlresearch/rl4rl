MECHANISM: Single-microbatch update densification

HYPOTHESIS: Using the available VRAM to train on one 192-sequence microbatch per step will yield `val_bpb < 0.995558` by providing roughly 33% more optimizer updates with near-baseline token throughput.

INTENDED_EDIT: Set the global batch equal to a larger 192-sequence device batch, eliminating gradient accumulation while reducing tokens per optimizer step from 524K to 393K.

EVIDENCE: The baseline reached 0.995558 using two accumulated 128-sequence microbatches and peaked at only 45,060 MB, leaving enough H100 memory to increase the device batch by 50%.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**17 # 393K tokens; one larger microbatch per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 192  # use spare H100 memory to eliminate gradient accumulation
>>>>>>> REPLACE