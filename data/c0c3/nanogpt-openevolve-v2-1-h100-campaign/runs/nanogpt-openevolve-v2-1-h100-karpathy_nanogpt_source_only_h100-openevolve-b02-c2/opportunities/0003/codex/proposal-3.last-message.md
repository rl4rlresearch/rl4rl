MECHANISM: Single-microbatch optimizer cadence

HYPOTHESIS: Halving the global batch to 262K tokens will lower val_bpb below 0.995558 by nearly doubling parameter-update opportunities while retaining the baseline architecture, attention windows, and efficient device batch.

INTENDED_EDIT: Set the total batch equal to one 128×2048-token device batch, eliminating gradient accumulation without changing per-forward execution.

EVIDENCE: The 497.0M-token baseline achieved 0.995558 in only 948 updates, while both attention execution changes reduced token throughput and worsened validation; this tests update frequency while preserving the successful attention implementation.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step; one device batch
>>>>>>> REPLACE