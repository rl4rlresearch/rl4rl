MECHANISM: Local batch-size response-surface refinement

HYPOTHESIS: A 116-sequence, 237,568-token single-pass batch will reduce val_bpb below 0.985757 by moving toward the roughly 117-sequence minimum implied by the measured 108-, 112-, and 128-sequence results.

INTENDED_EDIT: Increase both total and device batch sizes from 112 to 116 sequences while preserving one optimizer update per forward/backward pass.

EVIDENCE: The 108-, 112-, and 128-sequence batches achieved val_bpb 0.986047, 0.985757, and 0.986162; quadratic interpolation of these local measurements places the estimated minimum near 117 sequences.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 7 * 2**15 # 229,376 tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 29 * 2**13 # 237,568 tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 112  # single 229,376-token forward/backward pass
=======
DEVICE_BATCH_SIZE = 116  # single 237,568-token forward/backward pass
>>>>>>> REPLACE