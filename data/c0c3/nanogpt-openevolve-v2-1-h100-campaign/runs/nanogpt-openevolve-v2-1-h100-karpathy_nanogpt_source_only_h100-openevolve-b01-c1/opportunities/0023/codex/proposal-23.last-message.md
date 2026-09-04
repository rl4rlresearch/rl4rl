MECHANISM: Intermediate-batch optimization cadence

HYPOTHESIS: A 196K-token batch will balance the improved update frequency seen at 262K tokens against the throughput and optimization regression at 131K tokens, lowering val_bpb below 0.985493.

INTENDED_EDIT: Set both total and device batch size to 96 sequences (196,608 tokens), preserving one microbatch per optimizer step and the successful 0.6 warmdown schedule.

EVIDENCE: Reducing batch size from 524K to 262K improved val_bpb from 0.992286 to 0.986167, while reducing it further to 131K regressed to 0.991527; testing the midpoint directly probes the apparent interior optimum.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 96 * 2048 # ~196K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 96   # per-device batch size; one microbatch per optimizer step
>>>>>>> REPLACE