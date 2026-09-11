MECHANISM: Single-microbatch 320K-token updates

HYPOTHESIS: Reducing the single-microbatch update from 393,216 to 327,680 tokens will provide 20% more whole-model updates per token while retaining sufficient H100 occupancy, lowering val_bpb below 0.990146.

INTENDED_EDIT: Set both total and device batch sizes to 160 sequences, preserving one microbatch per optimizer step and all proven architecture, optimizer, and schedule settings.

EVIDENCE: The 393K single-microbatch design improved val_bpb from 0.994364 to 0.990146 at essentially unchanged training volume, while increasing steps from 939 to 1246; this strongly motivates testing a further update-frequency increase.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**17 # 393K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 5 * 2**16 # 320K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 192  # one larger microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 160  # one microbatch per optimizer step
>>>>>>> REPLACE