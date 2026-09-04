MECHANISM: Incremental leading-boundary frame pruning

HYPOTHESIS: Processing frames 2–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.2% versus the 31-step design.

INTENDED_EDIT: Omit frame 1 in addition to frame 0, producing an explicit 30-step causal schedule while preserving model capacity and training.

EVIDENCE: The 31-step model achieved 85.40% accuracy; prior 30-step attempts either timed out or accidentally reproduced the 31-step implementation, so the temporal-compression boundary remains unresolved.

<<<<<<< SEARCH
        return list(range(1, available_frames))
=======
        return list(range(2, available_frames))
>>>>>>> REPLACE