MECHANISM: Further throughput-preserving update densification

HYPOTHESIS: Halving the device and optimizer batch to 64 sequences will approximately double update count while retaining enough of the 128-sequence design’s 489.9M-token throughput to achieve `val_bpb < 0.987111`.

INTENDED_EDIT: Reduce both the global token batch and per-device sequence batch by half, preserving one microbatch per optimizer step.

EVIDENCE: Eliminating accumulation with a 128-sequence batch improved `val_bpb` from 0.995558 to 0.987111 while sustaining 489.9M tokens; this motivates testing whether another halving provides further update-density gains.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step; one efficient 128-sequence microbatch
=======
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step; one 64-sequence microbatch
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 64   # per-device batch size; matches one optimizer step
>>>>>>> REPLACE