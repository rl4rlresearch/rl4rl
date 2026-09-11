MECHANISM: One-step uniform temporal pruning

HYPOTHESIS: A 65-unit GRU processing 28 uniformly spaced frames will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 3.4% versus the verified 29-step model.

INTENDED_EDIT: Reduce the uniform full-utterance frame schedule from 29 to 28 causal recurrent steps while preserving recurrent width, temporal readout, and training procedure.

EVIDENCE: The 65-unit model achieved 86.01% accuracy at 29 steps, improving on the qualifying 30-step result despite using one fewer frame; this supports testing the next single-step reduction while retaining the recurrent capacity that succeeded where 64-unit variants failed.

<<<<<<< SEARCH
        target_steps = min(29, available_frames)
=======
        target_steps = min(28, available_frames)
>>>>>>> REPLACE