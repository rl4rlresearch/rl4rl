MECHANISM: Fine-grained linear warmdown interpolation

HYPOTHESIS: A 52% warmdown will preserve roughly 511M-token throughput and reduce val_bpb below 0.993287 by placing the decay length near the bracketed optimum between the tested 50% and 60% schedules.

INTENDED_EDIT: Shorten the current linear warmdown from 60% to 52% of the training window while preserving the proven architecture, batch size, and optimizer.

EVIDENCE: The 50% warmdown achieved 0.993287 val_bpb, while 40% and 60% regressed to 0.993995 and 0.993714 at nearly identical throughput; the asymmetric results motivate a narrow interpolation slightly above 50%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.52   # fraction of time budget for LR warmdown
>>>>>>> REPLACE