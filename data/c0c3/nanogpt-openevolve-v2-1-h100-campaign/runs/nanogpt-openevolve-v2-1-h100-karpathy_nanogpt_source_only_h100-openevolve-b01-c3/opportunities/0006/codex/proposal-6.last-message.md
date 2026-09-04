MECHANISM: Optimizer-overhead amortization with a three-microbatch global batch

HYPOTHESIS: Increasing the global batch to 786,432 tokens will amortize Muon’s per-update cost, process more than 497M tokens in five minutes, and reduce val_bpb below 0.995558 despite fewer optimizer steps.

INTENDED_EDIT: Increase gradient accumulation from one to three 128-sequence microbatches while preserving the proven SSSL architecture, five-step Muon refinement, and linear warmdown.

EVIDENCE: The 524K-token batch achieved 0.995558 and 497.0M tokens, whereas the 262K batch incurred more optimizer updates, processed only 393.2M tokens, and worsened to 0.998629; this indicates that further amortizing optimizer overhead is the clearest untested direction.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**18 # ~786K tokens per optimizer step
>>>>>>> REPLACE