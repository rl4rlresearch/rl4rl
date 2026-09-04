MECHANISM: Finer-grained stochastic optimization

HYPOTHESIS: Halving the global batch will preserve at least 465M-token throughput while roughly doubling optimizer updates, reducing val_bpb below 0.994296.

INTENDED_EDIT: Reduce the global batch from 524,288 to 262,144 tokens while retaining the proven 4.375× MLP, device batch, learning rates, and time-based schedules.

EVIDENCE: Increasing the batch to 786,432 reduced updates from 948 to 632 and worsened val_bpb from 0.995558 to 1.012938 at comparable token exposure, indicating that update frequency is valuable in this fixed-time regime.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
>>>>>>> REPLACE