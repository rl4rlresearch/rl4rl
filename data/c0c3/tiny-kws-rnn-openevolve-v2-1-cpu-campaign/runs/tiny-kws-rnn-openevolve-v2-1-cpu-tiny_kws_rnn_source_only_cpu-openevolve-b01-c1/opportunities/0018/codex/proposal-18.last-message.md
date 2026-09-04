MECHANISM: Incremental leading-boundary frame pruning

HYPOTHESIS: Processing frames 4–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.4% versus the 29-step design.

INTENDED_EDIT: Omit frame 3 in addition to frames 0–2, producing a 28-step causal schedule without changing model capacity or training.

EVIDENCE: Progressive leading-frame pruning from 32 to 29 steps preserved or improved accuracy, with the 29-step model reaching 85.77%; this supports testing the smallest further temporal reduction.

<<<<<<< SEARCH
        return list(range(3, available_frames))
=======
        return list(range(4, available_frames))
>>>>>>> REPLACE