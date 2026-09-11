MECHANISM: Upper-side optimizer-batch midpoint refinement

HYPOTHESIS: A 253,952-token optimizer batch will retain roughly 518M training tokens while lowering val_bpb below 0.985506.

INTENDED_EDIT: Increase the single microbatch from 112 to 124 sequences, preserving the architecture, optimizer hyperparameters, and 60% linear warmdown.

EVIDENCE: The 245,760-token batch achieved the best val_bpb of 0.985506, while 229,376 reached 0.985754 and 262,144 reached 0.985657; 253,952 tests the midpoint of the tighter upper bracket around the empirical optimum.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 7 * 2**15 # ~229K tokens; one 112-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 31 * 2**13 # ~254K tokens; one 124-sequence microbatch per step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 112  # one microbatch per 229,376-token optimizer step
=======
DEVICE_BATCH_SIZE = 124  # one microbatch per 253,952-token optimizer step
>>>>>>> REPLACE