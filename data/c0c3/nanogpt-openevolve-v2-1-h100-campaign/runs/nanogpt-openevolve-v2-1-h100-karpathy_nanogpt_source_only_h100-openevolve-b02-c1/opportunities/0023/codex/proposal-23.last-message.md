MECHANISM: Single-microbatch 176K-token updates

HYPOTHESIS: An 180,224-token update will balance the update-frequency benefit of smaller batches against the throughput and gradient-noise regression observed at 163,840 tokens, reducing val_bpb below 0.986971.

INTENDED_EDIT: Set both total and device batch sizes to 88 sequences, preserving one microbatch per optimizer step and all proven architecture, optimizer, and schedule settings.

EVIDENCE: Reducing updates from 262,144 to 196,608 tokens improved val_bpb from 0.987386 to 0.986971, but a further reduction to 163,840 regressed to 0.988167; testing the midpoint directly probes the apparent optimum.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**16 # 192K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 11 * 2**14 # 176K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 96   # one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 88   # one microbatch per optimizer step
>>>>>>> REPLACE