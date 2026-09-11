MECHANISM: Single-microbatch 256K-token updates

HYPOTHESIS: Reducing the update batch from 327,680 to 262,144 tokens will provide 25% more whole-model updates per token while preserving the previously demonstrated throughput of 128-sequence microbatches, lowering val_bpb below 0.988487.

INTENDED_EDIT: Set both total and device batch sizes to 128 sequences, retaining one microbatch per optimizer step and all proven architecture, optimizer, and schedule settings.

EVIDENCE: Reducing single-microbatch updates from 393,216 to 327,680 tokens improved val_bpb from 0.990146 to 0.988487, while prior 128-sequence runs demonstrated roughly 492M-token throughput; this motivates continuing the update-frequency trend at a known-efficient device batch size.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 5 * 2**16 # 320K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 4 * 2**16 # 256K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 160  # one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 128  # one microbatch per optimizer step
>>>>>>> REPLACE