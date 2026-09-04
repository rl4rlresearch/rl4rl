MECHANISM: Five-percent Muon trajectory-length reduction

HYPOTHESIS: Using the proven 258,048-token batch with `MATRIX_LR=0.038` will retain roughly 522M training tokens and lower val_bpb below 0.985181.

INTENDED_EDIT: Restore the best 126-sequence microbatch and reduce only the Muon matrix learning rate by 5%, preserving five polar iterations and all other settings.

EVIDENCE: The 258,048-token design achieved the best val_bpb of 0.985181, while increasing `MATRIX_LR` by 5% regressed to 0.985838; testing the symmetric lower side is the most direct remaining learning-rate bracket.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 127 * 2**11 # ~260K tokens; one 127-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 63 * 2**12 # ~258K tokens; one 126-sequence microbatch per step
>>>>>>> REPLACE

<<<<<<< SEARCH
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
=======
MATRIX_LR = 0.038       # 5% lower Muon learning rate at the best batch size
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 127  # one microbatch per 260,096-token optimizer step
=======
DEVICE_BATCH_SIZE = 126  # one microbatch per 258,048-token optimizer step
>>>>>>> REPLACE