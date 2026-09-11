MECHANISM: Incremental leading-boundary frame pruning

HYPOTHESIS: Processing frames 3–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.3% versus the 30-step design.

INTENDED_EDIT: Omit frame 2 in addition to frames 0–1, producing a 29-step causal schedule without changing model capacity or training.

EVIDENCE: The 92-unit GRU retained 85.52% accuracy after progressive leading-frame pruning reduced execution from 32 to 30 steps, indicating that the earliest boundary frames are lower-value and motivating the smallest remaining temporal reduction.

<<<<<<< SEARCH
        return list(range(2, available_frames))
=======
        return list(range(3, available_frames))
>>>>>>> REPLACE