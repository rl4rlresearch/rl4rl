MECHANISM: Single-microbatch 384K-token updates

HYPOTHESIS: Using batch size 192 with one 393,216-token microbatch per optimizer step will provide 33% more whole-model updates per token while maintaining high GPU occupancy, reducing val_bpb below 0.994364.

INTENDED_EDIT: Increase the device batch from 128 to 192 and reduce total batch size to exactly one device microbatch, preserving the best architecture and optimizer settings.

EVIDENCE: Raising only the lexical expert learning rate regressed val_bpb to 0.995174, while the unchanged channel-wise expert remains best at 0.994364; this motivates increasing update frequency for the whole proven model instead of further accelerating or expanding the lexical path.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**17 # 393K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 192  # one larger microbatch per optimizer step
>>>>>>> REPLACE