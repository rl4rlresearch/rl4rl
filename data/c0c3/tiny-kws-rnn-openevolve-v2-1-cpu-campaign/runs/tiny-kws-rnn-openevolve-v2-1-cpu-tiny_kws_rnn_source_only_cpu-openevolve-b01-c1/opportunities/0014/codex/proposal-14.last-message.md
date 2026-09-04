MECHANISM: Incremental leading-boundary frame pruning

HYPOTHESIS: Processing frames 2–31 with the verified 92-unit GRU will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.2% versus the 31-step design.

INTENDED_EDIT: Omit one additional earliest frame, reducing the causal schedule from 31 to 30 recurrent steps while preserving model capacity and training.

EVIDENCE: The 31-step model achieved 85.40% accuracy; the intended 30-step test timed out and the subsequent edit reproduced the 31-step implementation, so this compression boundary remains unresolved.

<<<<<<< SEARCH
        return list(range(1, available_frames))
=======
        return list(range(2, available_frames))
>>>>>>> REPLACE