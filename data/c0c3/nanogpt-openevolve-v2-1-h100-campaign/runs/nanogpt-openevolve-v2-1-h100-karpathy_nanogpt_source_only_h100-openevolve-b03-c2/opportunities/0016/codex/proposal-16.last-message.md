MECHANISM: Single-microbatch updates with a 55% linear refinement tail

HYPOTHESIS: Restoring the jointly verified 262K-token update geometry and 55% linear warmdown will reduce val_bpb below the current design’s 0.995558 and reproduce the strongest available result near 0.986676.

INTENDED_EDIT: Halve the global batch so every optimizer step uses one 128-sequence microbatch, and begin linear cooldown after 45% of training.

EVIDENCE: Reference Design 3 achieved the best available val_bpb of 0.986676 with 491.5M tokens, while the current 524K-token batch reached only 0.995558; adjacent 54% and 56% cooldowns were also worse than 55%.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens, one microbatch per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.55   # linear refinement over the final 55% of training
>>>>>>> REPLACE