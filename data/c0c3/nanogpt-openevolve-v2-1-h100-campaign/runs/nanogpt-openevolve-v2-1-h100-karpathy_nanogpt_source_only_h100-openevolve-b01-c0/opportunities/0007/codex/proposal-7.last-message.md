MECHANISM: Single-microbatch training with increased device occupancy

HYPOTHESIS: A 393,216-token batch processed as one 192-sequence microbatch will beat 0.993324 val_bpb by eliminating gradient accumulation and producing roughly 30% more optimizer updates while preserving most token throughput.

INTENDED_EDIT: Increase the device batch from 128 to 192 and reduce the total batch to exactly one device microbatch.

EVIDENCE: The best 512-token SSSL design uses two accumulated 262,144-token microbatches, reaches only 981 updates, and peaks at 45,060 MB VRAM, leaving enough H100 memory to test a larger single microbatch without changing the successful model or attention layout.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**17 # ~393K tokens per optimizer step; one 192-sequence microbatch
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 192  # use available H100 memory to eliminate gradient accumulation
>>>>>>> REPLACE