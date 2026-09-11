MECHANISM: Local batch-size response-surface refinement

HYPOTHESIS: A 113-sequence, 231,424-token single-pass batch will reduce val_bpb below 0.985757 by matching the approximately 113-sequence minimum implied by the nearest measured batch sizes.

INTENDED_EDIT: Increase both total and device batch sizes from 112 to 113 sequences while preserving one optimizer update per forward/backward pass.

EVIDENCE: The 108-, 112-, and 116-sequence batches achieved val_bpb values of 0.986047, 0.985757, and 0.985866; quadratic interpolation of these closest measurements estimates the minimum near 113 sequences.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 7 * 2**15 # 229,376 tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 113 * 2**11 # 231,424 tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 112  # single 229,376-token forward/backward pass
=======
DEVICE_BATCH_SIZE = 113  # single 231,424-token forward/backward pass
>>>>>>> REPLACE