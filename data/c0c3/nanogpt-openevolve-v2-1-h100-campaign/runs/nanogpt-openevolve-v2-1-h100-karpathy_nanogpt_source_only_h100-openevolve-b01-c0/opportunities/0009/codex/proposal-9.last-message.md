MECHANISM: 131K-token single-microbatch optimizer cadence

HYPOTHESIS: Halving the optimizer batch to 131,072 tokens will beat 0.985318 val_bpb by doubling update frequency, provided the 64-sequence microbatch retains enough throughput to process a comparable token count.

INTENDED_EDIT: Reduce both the total batch and device batch by half, preserving one microbatch per optimizer step and the proven 512-token SSSL architecture.

EVIDENCE: Eliminating two-way accumulation at 262,144 tokens nearly doubled updates from 981 to 1933 and improved val_bpb from 0.993324 to 0.985318 while retaining 506.7M training tokens, motivating the adjacent smaller-batch test.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 64   # per-device batch size; one microbatch per optimizer step
>>>>>>> REPLACE