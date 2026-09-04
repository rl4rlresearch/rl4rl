MECHANISM: More frequent stochastic optimizer updates

HYPOTHESIS: Halving the effective batch will beat val_bpb 0.995558 by nearly doubling optimizer updates within five minutes while retaining most token throughput.

INTENDED_EDIT: Reduce total batch size from 524K to 262K tokens, changing gradient accumulation from two microbatches to one.

EVIDENCE: The 497.0M-token baseline remains best after attention-pattern experiments; with only 948 updates, optimization frequency is the clearest untested lever.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
>>>>>>> REPLACE