MECHANISM: Incremental leading-boundary frame pruning

HYPOTHESIS: Processing frames 2–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.2% versus the 31-step design.

INTENDED_EDIT: Omit one additional earliest frame, reducing the causal schedule from 31 to 30 recurrent steps while leaving model capacity and training unchanged.

EVIDENCE: Omitting frame 0 preserved 85.40% accuracy and reduced inference cost versus the 32-step model; pruning one more likely low-information boundary frame is the smallest informative test of the temporal-compression limit.

<<<<<<< SEARCH
        return list(range(1, available_frames))
=======
        return list(range(2, available_frames))
>>>>>>> REPLACE