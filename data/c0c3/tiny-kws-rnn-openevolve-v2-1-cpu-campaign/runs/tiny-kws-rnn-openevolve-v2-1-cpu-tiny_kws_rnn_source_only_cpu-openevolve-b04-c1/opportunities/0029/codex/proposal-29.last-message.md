MECHANISM: Symmetric boundary-frame temporal pruning

HYPOTHESIS: Processing the 28 contiguous central frames (indices 2–29) will retain at least 85% validation accuracy while reducing recurrent steps and dense MACs by approximately 3.4% versus the verified 29-step model.

INTENDED_EDIT: Reduce the causal schedule from 29 to 28 steps, removing the latest remaining boundary frame while preserving the 58-unit GRU, temporal pooling, and training procedure.

EVIDENCE: The central 30-step model achieved 85.77% accuracy and the 29-step model achieved 85.64%; both passed, indicating that boundary-frame pruning has preserved accuracy so far and motivating the complementary boundary removal.

<<<<<<< SEARCH
        steps = min(29, available_frames)
=======
        steps = min(28, available_frames)
>>>>>>> REPLACE