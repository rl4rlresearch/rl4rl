MECHANISM: Single-microbatch half-window SSSL training

HYPOTHESIS: Restoring the proven 1024-token short windows while reducing the optimizer batch to one 128-sequence microbatch will retain high token throughput and provide substantially more updates, lowering val_bpb below 0.995558.

INTENDED_EDIT: Restore half-context attention and use a 262,144-token batch with no gradient accumulation.

EVIDENCE: Reference Design 1 achieved 0.995558 with half-context SSSL and 128-sequence microbatches. Reference Design 2 suggests more frequent single-microbatch updates preserve quality despite fewer tokens, but its 192-sequence microbatch reduced throughput; retaining the efficient 128-sequence microbatch isolates the update-frequency benefit.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = long_window // 2
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # 262K tokens; one 128-sequence microbatch per step
>>>>>>> REPLACE