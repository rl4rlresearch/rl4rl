MECHANISM: Intermediate-frequency single-microbatch optimization

HYPOTHESIS: A 196,608-token batch will lower val_bpb below 0.986162 by balancing the better update frequency of smaller batches against the optimization and throughput regression observed at 131,072 tokens.

INTENDED_EDIT: Set both total and device batch sizes to a 96-sequence, 196,608-token single-pass optimizer step.

EVIDENCE: Reducing the batch from 524,288 to 262,144 improved val_bpb from 0.993365 to 0.986162, but reducing it further to 131,072 regressed to 0.990069; this directly tests the midpoint between the two measured settings.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**16 # 196,608 tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 96   # single 196,608-token forward/backward pass
>>>>>>> REPLACE