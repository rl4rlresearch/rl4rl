MECHANISM: Delayed linear learning-rate warmdown

HYPOTHESIS: Restoring the proven 524K-token, 128-sequence configuration while delaying warmdown from 50% to 60% elapsed time will retain roughly 497M-token throughput and lower val_bpb below 0.995558 by applying peak learning rates to more training tokens while preserving a 120-second cooldown.

INTENDED_EDIT: Restore Reference Design 1’s batch configuration and shorten LR warmdown from half to 40% of the training window.

EVIDENCE: Reference Design 1 achieved the best val_bpb, 0.995558, with 497.0M tokens and 128-sequence microbatches; smaller or irregular batches produced fewer tokens without improving validation, motivating an optimization-schedule test on the proven configuration.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**17 # 393K tokens; one 192-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 2**19     # 524K tokens; two 128-sequence microbatches per step
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.4    # delay decay until 60% progress, then warm down linearly
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 192  # use available H100 memory to eliminate accumulation
=======
DEVICE_BATCH_SIZE = 128  # proven high-throughput H100 microbatch
>>>>>>> REPLACE