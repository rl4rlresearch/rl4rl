MECHANISM: Empirical batch-size interpolation

HYPOTHESIS: A 229,376-token single-pass batch will lower val_bpb below 0.985961 by moving toward the approximately 111-sequence optimum implied by the measured 64-, 96-, and 128-sequence results.

INTENDED_EDIT: Increase both total and device batch sizes from 96 to 112 sequences while preserving one optimizer update per forward/backward pass.

EVIDENCE: The 131,072-, 196,608-, and 262,144-token batches achieved val_bpb 0.990069, 0.985961, and 0.986162 respectively; quadratic interpolation of these evenly spaced measurements places the estimated minimum near 111 sequences, directly motivating a 112-sequence test.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**16 # 196,608 tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 7 * 2**15 # 229,376 tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 96   # single 196,608-token forward/backward pass
=======
DEVICE_BATCH_SIZE = 112  # single 229,376-token forward/backward pass
>>>>>>> REPLACE