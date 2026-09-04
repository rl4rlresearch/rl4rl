MECHANISM: Higher-frequency optimizer updates via halved global batch

HYPOTHESIS: Halving the optimizer batch to 262K tokens will approximately double the number of updates while retaining at least 500M training tokens, lowering val_bpb below 0.992358.

INTENDED_EDIT: Reduce gradient accumulation from two 128-sequence microbatches per optimizer step to one, preserving the model, device batch, optimizer hyperparameters, and proven 60% linear warmdown.

EVIDENCE: The current 256-token, 60%-warmdown design is best at 0.992358 and processes 528.5M tokens, while further attention-window, schedule, and global-layer refinements regressed; batch/update frequency is therefore an informative orthogonal optimization axis.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens; two 128-sequence microbatches per step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens; one 128-sequence microbatch per step
>>>>>>> REPLACE