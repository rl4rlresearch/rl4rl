MECHANISM: Higher-frequency single-microbatch optimization

HYPOTHESIS: Halving the optimizer batch to 262,144 tokens will lower val_bpb below 0.993365 by roughly doubling parameter updates per trained token while retaining the proven 128-sequence device batch.

INTENDED_EDIT: Reduce total batch size from 524,288 to 262,144 tokens, changing gradient accumulation from two microbatches to one.

EVIDENCE: The best design reached 0.993365 with only 977 optimizer updates, while the attempted 256-sequence single-pass batch could not be verified; this obtains single-pass updates without increasing the known-fitting device batch.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
>>>>>>> REPLACE