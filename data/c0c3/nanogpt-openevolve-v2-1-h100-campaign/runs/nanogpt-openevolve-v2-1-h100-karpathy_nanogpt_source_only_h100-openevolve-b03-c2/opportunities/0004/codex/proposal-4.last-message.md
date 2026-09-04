MECHANISM: Single-microbatch 197K-token updates

HYPOTHESIS: Reducing the global batch from 262K to 197K tokens while using the already efficient 96-sequence microbatch will increase optimizer updates by roughly one-third and lower val_bpb below 0.98713 without materially reducing token throughput.

INTENDED_EDIT: Set both the global batch and device batch so each optimizer step processes one 96-sequence microbatch.

EVIDENCE: The 96-sequence design sustained 490.7M tokens and 39.18% MFU, while reducing update size from 393K to 262K tokens improved val_bpb from 0.990599 to 0.98713 at similar throughput.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 96   # one microbatch per optimizer step
>>>>>>> REPLACE