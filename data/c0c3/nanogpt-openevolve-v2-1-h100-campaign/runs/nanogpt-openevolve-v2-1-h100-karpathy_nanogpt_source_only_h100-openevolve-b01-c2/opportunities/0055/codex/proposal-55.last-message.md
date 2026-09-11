MECHANISM: Parabolic warmdown-onset interpolation

HYPOTHESIS: A 58.65% zero-ending linear warmdown with the proven 258,048-token batch will retain roughly 522M training tokens and lower `val_bpb` below 0.985148.

INTENDED_EDIT: Restore the best 126-sequence microbatch and set warmdown to 58.65%, preserving all other architecture and optimizer settings.

EVIDENCE: At identical batch size, 57%, 58.5%, and 60% warmdowns produced `val_bpb` 0.985198, 0.985148, and 0.985181; parabolic interpolation places the estimated minimum near 58.65%.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 31 * 2**13 # ~254K tokens; one 124-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 63 * 2**12 # ~258K tokens; one 126-sequence microbatch per step
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # begin linear LR warmdown after 40% of the time budget
=======
WARMDOWN_RATIO = 0.5865 # interpolated optimum: begin linear warmdown after 41.35% of the time budget
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 124  # one microbatch per 253,952-token optimizer step
=======
DEVICE_BATCH_SIZE = 126  # one microbatch per 258,048-token optimizer step
>>>>>>> REPLACE