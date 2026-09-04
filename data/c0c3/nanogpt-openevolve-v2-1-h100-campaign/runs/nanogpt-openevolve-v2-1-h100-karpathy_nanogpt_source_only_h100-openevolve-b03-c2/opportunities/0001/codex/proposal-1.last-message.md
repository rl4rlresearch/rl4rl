MECHANISM: Single-microbatch 393K-token updates

HYPOTHESIS: A 192-sequence microbatch will fit the H100’s remaining memory, eliminate gradient accumulation, and provide more optimizer updates per token, reducing val_bpb within five minutes.

INTENDED_EDIT: Reduce the global batch from 524K to 393K tokens and increase the device batch from 128 to 192, changing each update from two microbatches to one.

EVIDENCE: The baseline reached val_bpb 0.995558 with 948 updates and 45,060 MB peak VRAM; substantial memory headroom supports a 1.5× larger microbatch, while the smaller global batch increases update frequency.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**17 # ~393K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 192  # one microbatch per optimizer step
>>>>>>> REPLACE