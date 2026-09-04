MECHANISM: Single-microbatch 262K-token updates

HYPOTHESIS: Halving the baseline global batch while retaining its efficient 128-sequence microbatch will increase update frequency enough to beat the best observed val_bpb of 0.990599 without the throughput collapse of the 192-sequence design.

INTENDED_EDIT: Reduce each optimizer update from 524K tokens over two microbatches to 262K tokens in one 128-sequence microbatch.

EVIDENCE: The 393K-token, two-microbatch design improved val_bpb from 0.995558 to 0.990599 while preserving 39.18% MFU and 490.7M tokens; the 192-sequence single-microbatch result shows that increasing device batch size harms throughput, motivating smaller updates using the baseline’s efficient device batch.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 128  # one microbatch per optimizer step
>>>>>>> REPLACE