MECHANISM: Throughput-preserving update densification

HYPOTHESIS: A single 128-sequence microbatch per optimizer step will retain the baseline’s roughly 497M-token throughput while doubling optimizer updates, yielding `val_bpb < 0.995558`.

INTENDED_EDIT: Halve the global batch from 524K to 262K tokens, matching the existing 128-sequence device batch and eliminating gradient accumulation.

EVIDENCE: The 192-sequence single-microbatch trial regressed to `val_bpb 1.003716` because throughput fell from 497.0M to 395.2M tokens; the baseline demonstrates that 128-sequence microbatches efficiently sustain the higher throughput.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step; one efficient 128-sequence microbatch
>>>>>>> REPLACE