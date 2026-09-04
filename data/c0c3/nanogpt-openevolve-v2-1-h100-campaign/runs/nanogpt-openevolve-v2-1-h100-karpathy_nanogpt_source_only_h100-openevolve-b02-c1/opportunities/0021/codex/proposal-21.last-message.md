MECHANISM: Single-microbatch 192K-token updates

HYPOTHESIS: Reducing the update batch from 262,144 to 196,608 tokens will provide 33% more whole-model updates per token while retaining adequate H100 occupancy, lowering val_bpb below 0.987386.

INTENDED_EDIT: Set both total and device batch sizes to 96 sequences, preserving one microbatch per optimizer step and all proven architecture, optimizer, and schedule settings.

EVIDENCE: Successive single-microbatch reductions from 393,216 to 327,680 to 262,144 tokens improved val_bpb from 0.990146 to 0.988487 to 0.987386 despite modest throughput losses, strongly motivating another update-frequency increase.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 4 * 2**16 # 256K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**16 # 192K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 96   # one microbatch per optimizer step
>>>>>>> REPLACE