MECHANISM: Continued symmetric boundary-frame pruning

HYPOTHESIS: Processing the 26 contiguous central frames (indices 3–28) will retain at least 85% validation accuracy while reducing recurrent steps by 3.7% and total dense inference MACs to approximately 288.7M.

INTENDED_EDIT: Reduce the causal schedule from 27 to 26 steps, removing the latest remaining boundary frame while preserving the verified 58-unit GRU, temporal pooling, and training procedure.

EVIDENCE: The contiguous 28-step and 27-step models achieved 86.99% and 86.38% accuracy respectively, and every boundary-pruning design from 31 through 27 steps passed; the current 1.38-point margin supports testing the complementary boundary removal.

<<<<<<< SEARCH
        steps = min(27, available_frames)
=======
        steps = min(26, available_frames)
>>>>>>> REPLACE