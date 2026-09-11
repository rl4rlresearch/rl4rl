MECHANISM: Single-microbatch 160K-token updates

HYPOTHESIS: Reducing the update batch from 196,608 to 163,840 tokens will provide 20% more whole-model updates per token while retaining adequate H100 occupancy, lowering val_bpb below 0.986971.

INTENDED_EDIT: Set both total and device batch sizes to 80 sequences, preserving one microbatch per optimizer step and all proven architecture, optimizer, and schedule settings.

EVIDENCE: Successive single-microbatch reductions from 393,216 to 327,680 to 262,144 to 196,608 tokens consistently improved val_bpb from 0.990146 to 0.988487 to 0.987386 to 0.986971, while throughput declined only modestly; a smaller incremental reduction tests whether the update-frequency benefit continues near the apparent optimum.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**16 # 192K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 5 * 2**15 # 160K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 96   # one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 80   # one microbatch per optimizer step
>>>>>>> REPLACE