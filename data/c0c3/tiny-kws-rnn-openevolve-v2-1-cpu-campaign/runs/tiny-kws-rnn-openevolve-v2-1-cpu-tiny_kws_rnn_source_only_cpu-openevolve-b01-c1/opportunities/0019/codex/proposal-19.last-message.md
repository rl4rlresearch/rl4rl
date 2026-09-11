MECHANISM: Incremental leading-boundary frame pruning

HYPOTHESIS: Processing frames 5–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.6% versus the 28-step design.

INTENDED_EDIT: Omit frame 4 in addition to frames 0–3, producing a 27-step causal schedule without changing model capacity or training.

EVIDENCE: Progressive leading-frame pruning from 32 to 28 steps has consistently retained at least 85% accuracy; the current 28-step model achieves 85.40%, making one additional leading-frame removal the smallest informative test of the temporal-compression boundary.

<<<<<<< SEARCH
        return list(range(4, available_frames))
=======
        return list(range(5, available_frames))
>>>>>>> REPLACE