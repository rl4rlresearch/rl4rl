MECHANISM: Tight upper-bracket optimizer-batch refinement

HYPOTHESIS: A 260,096-token optimizer batch will retain roughly 520M training tokens while lowering val_bpb below 0.985181.

INTENDED_EDIT: Use one 127-sequence microbatch per optimizer step while preserving the proven architecture, optimizer hyperparameters, and 60% linear warmdown.

EVIDENCE: The 258,048-token batch achieved the best val_bpb of 0.985181, improving over 253,952 tokens at 0.985215, while 262,144 tokens regressed to 0.985657; 260,096 tokens tests the exact midpoint of this tight bracket.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens; one 64-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 127 * 2**11 # ~260K tokens; one 127-sequence microbatch per step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 64   # efficient per-device batch size for 131K-token steps
=======
DEVICE_BATCH_SIZE = 127  # one microbatch per 260,096-token optimizer step
>>>>>>> REPLACE