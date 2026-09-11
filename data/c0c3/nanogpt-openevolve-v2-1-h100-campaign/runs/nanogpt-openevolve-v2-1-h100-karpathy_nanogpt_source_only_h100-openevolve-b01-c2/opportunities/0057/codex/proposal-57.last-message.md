MECHANISM: Microbatch-size boundary refinement

HYPOTHESIS: A 260,096-token optimizer batch with the best 58.5% zero-ending linear warmdown will retain roughly 523M training tokens and lower `val_bpb` below 0.985148.

INTENDED_EDIT: Increase the single-step microbatch from 126 to 127 sequences while preserving the best architecture, optimizer settings, learning rates, and warmdown schedule.

EVIDENCE: At the 58.5% warmdown, 258,048 tokens achieved the best `val_bpb` of 0.985148, while 262,144 tokens regressed to 0.985354; testing their unmeasured midpoint isolates the local batch-size optimum.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 63 * 2**12 # ~258K tokens; one 126-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 127 * 2**11 # ~260K tokens; one 127-sequence microbatch per step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 126  # one microbatch per 258,048-token optimizer step
=======
DEVICE_BATCH_SIZE = 127  # one microbatch per 260,096-token optimizer step
>>>>>>> REPLACE