MECHANISM: Single-microbatch 184K-token updates

HYPOTHESIS: A 188,416-token update will better balance update frequency and GPU throughput within the apparent 180K–197K optimum, reducing val_bpb below 0.986967.

INTENDED_EDIT: Set both total and device batch sizes to 92 sequences, preserving one microbatch per optimizer step and all proven model and optimizer settings.

EVIDENCE: The 180,224-token design achieved 0.986967 and the 196,608-token design achieved a nearly identical 0.986971, while 163,840 tokens regressed to 0.988167; testing their midpoint refines the upper side of the apparent optimum.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 11 * 2**14 # 176K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 23 * 2**13 # 184K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 88   # one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 92   # one microbatch per optimizer step
>>>>>>> REPLACE