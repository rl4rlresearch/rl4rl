MECHANISM: Bracketed optimizer-batch refinement

HYPOTHESIS: A 229,376-token optimizer batch will retain at least 490M training tokens and reduce val_bpb below 0.985044.

INTENDED_EDIT: Increase the current batch to 112 sequences per single-microbatch optimizer step while preserving the best architecture, learning rates, and schedules.

EVIDENCE: The 196,608-token batch achieved the best val_bpb, 0.985044, while neighboring 131,072- and 262,144-token batches scored 0.989256 and 0.985237; their results bracket an estimated optimum near 225K tokens.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 7 * 2**15 # ~229K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 64   # one 131K-token microbatch per optimizer step
=======
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 112  # one 229K-token microbatch per optimizer step
>>>>>>> REPLACE