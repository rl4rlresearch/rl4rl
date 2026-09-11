MECHANISM: Continued boundary-frame temporal pruning

HYPOTHESIS: Processing the 25 contiguous near-central frames at indices 4–28 will retain at least 85% validation accuracy while reducing recurrent steps by 3.8% and total dense inference MACs to approximately 277.7M.

INTENDED_EDIT: Reduce the causal schedule from 26 to 25 steps by removing the earliest remaining boundary frame while preserving the verified 58-unit GRU, temporal pooling, and training procedure.

EVIDENCE: The 26-step central-window model achieved 86.01% accuracy, and every contiguous boundary-pruning design from 31 through 26 steps passed; its 1.01-point margin supports probing one additional boundary removal.

<<<<<<< SEARCH
        steps = min(26, available_frames)
=======
        steps = min(25, available_frames)
>>>>>>> REPLACE