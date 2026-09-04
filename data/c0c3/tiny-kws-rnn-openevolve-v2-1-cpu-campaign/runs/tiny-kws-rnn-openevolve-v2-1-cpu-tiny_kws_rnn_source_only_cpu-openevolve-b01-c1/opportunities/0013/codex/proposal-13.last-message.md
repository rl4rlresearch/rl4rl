MECHANISM: Incremental leading-boundary frame pruning

HYPOTHESIS: Processing frames 2–31 will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by approximately 3.2% versus the verified 31-step design.

INTENDED_EDIT: Omit the second-earliest frame, reducing the causal schedule from 31 to 30 steps without changing capacity or training.

EVIDENCE: The 31-step 92-unit GRU achieved 85.40% accuracy; the prior 30-step attempt timed out and therefore provided no accuracy evidence, making this smallest additional temporal reduction the unresolved boundary test.

<<<<<<< SEARCH
        return list(range(1, available_frames))
=======
        return list(range(2, available_frames))
>>>>>>> REPLACE