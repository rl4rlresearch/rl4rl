MECHANISM: Intermediate single-microbatch update frequency

HYPOTHESIS: A 196,608-token batch will beat 0.987174 by providing 33% more optimizer updates per token than the current design while avoiding the severe throughput loss observed at a 131,072-token batch.

INTENDED_EDIT: Set the effective and device batch to 96 sequences, preserving one microbatch per optimizer step.

EVIDENCE: Halving the batch from 524K to 262K preserved throughput and improved val_bpb from 0.995558 to 0.987174, while halving again to 131K reduced throughput from 489.9M to 366.9M tokens; the intermediate batch tests the tradeoff before that occupancy cliff.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 96   # per-device batch size (one microbatch per step)
>>>>>>> REPLACE