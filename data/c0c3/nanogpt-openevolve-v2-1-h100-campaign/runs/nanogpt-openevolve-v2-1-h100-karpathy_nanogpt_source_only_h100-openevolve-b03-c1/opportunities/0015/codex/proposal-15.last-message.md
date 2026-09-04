MECHANISM: Local batch-size response-surface refinement

HYPOTHESIS: A 108-sequence, 221,184-token single-pass batch will reduce val_bpb below 0.985757 by moving closer to the approximately 109-sequence minimum implied by the measured 96-, 112-, and 128-sequence results.

INTENDED_EDIT: Reduce both total and device batch sizes from 112 to 108 sequences while preserving one optimizer update per forward/backward pass.

EVIDENCE: Batch sizes of 96, 112, and 128 sequences produced val_bpb values of 0.985961, 0.985757, and 0.986162; quadratic interpolation of these equally spaced measurements estimates the local optimum near 109 sequences.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 7 * 2**15 # 229,376 tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 108 * 2048 # 221,184 tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 112  # single 229,376-token forward/backward pass
=======
DEVICE_BATCH_SIZE = 108  # single 221,184-token forward/backward pass
>>>>>>> REPLACE