MECHANISM: One-step uniform temporal pruning

HYPOTHESIS: A 65-unit GRU processing 27 uniformly spaced frames will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 3.6% versus the verified 28-step model.

INTENDED_EDIT: Reduce the uniform full-utterance frame schedule from 28 to 27 causal recurrent steps while preserving recurrent width, temporal readout, and training procedure.

EVIDENCE: The 65-unit model qualified at 28 steps with 85.77% accuracy after also qualifying at 29 and 30 steps; this supports the next minimal probe of temporal redundancy.

<<<<<<< SEARCH
        target_steps = min(28, available_frames)
=======
        target_steps = min(27, available_frames)
>>>>>>> REPLACE