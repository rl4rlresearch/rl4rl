MECHANISM: Single-microbatch 168K-token updates

HYPOTHESIS: A 172,032-token update will improve update frequency over the best 180,224-token design without incurring the gradient-noise regression observed at 163,840 tokens, reducing val_bpb below 0.986967.

INTENDED_EDIT: Set both total and device batch sizes to 84 sequences, preserving one microbatch per optimizer step and all proven architecture, optimizer, and schedule settings.

EVIDENCE: The 180,224-token design achieved the best val_bpb of 0.986967, while 163,840 tokens regressed to 0.988167; testing their exact midpoint most directly refines the lower side of the apparent optimum.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 11 * 2**14 # 176K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 21 * 2**13 # 168K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 88   # one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 84   # one microbatch per optimizer step
>>>>>>> REPLACE