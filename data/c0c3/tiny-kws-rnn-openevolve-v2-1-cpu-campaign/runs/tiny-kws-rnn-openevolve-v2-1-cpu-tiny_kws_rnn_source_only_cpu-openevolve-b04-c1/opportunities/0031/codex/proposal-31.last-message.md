MECHANISM: Continued boundary-frame temporal pruning

HYPOTHESIS: Processing 27 contiguous near-central frames (indices 3–29) will retain at least 85% validation accuracy while reducing recurrent steps by 3.6% and total dense inference MACs to approximately 299.8M.

INTENDED_EDIT: Reduce the causal schedule from 28 to 27 steps by removing the earliest remaining boundary frame, preserving the verified 58-unit GRU, pooling, and training procedure.

EVIDENCE: The 28-step model achieved 86.99% accuracy, a 1.99-point margin above threshold, after every contiguous boundary-pruning result from 31 through 28 steps passed; this supports probing one additional boundary frame without weakening recurrent capacity.

<<<<<<< SEARCH
        steps = min(28, available_frames)
=======
        steps = min(27, available_frames)
>>>>>>> REPLACE