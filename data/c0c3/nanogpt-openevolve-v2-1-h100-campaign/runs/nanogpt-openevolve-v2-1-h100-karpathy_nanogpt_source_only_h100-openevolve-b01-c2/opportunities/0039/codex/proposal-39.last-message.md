MECHANISM: Upper-bracket optimizer-batch midpoint refinement

HYPOTHESIS: A 258,048-token optimizer batch will retain roughly 518M training tokens while lowering val_bpb below 0.985215.

INTENDED_EDIT: Use one 126-sequence microbatch per optimizer step while preserving the architecture, optimizer hyperparameters, and 60% linear warmdown.

EVIDENCE: The 253,952-token batch achieved the best val_bpb of 0.985215, improving over 245,760 tokens at 0.985506, while 262,144 tokens regressed to 0.985657; 258,048 tokens tests the midpoint of this tight bracket.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens; one 96-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 63 * 2**12 # ~258K tokens; one 126-sequence microbatch per step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 96   # one microbatch per 196,608-token optimizer step
=======
DEVICE_BATCH_SIZE = 126  # one microbatch per 258,048-token optimizer step
>>>>>>> REPLACE